import type { ReasoningEffort } from "../chat/effort.ts";
import { runEdit } from "./askDara.ts";
import {
  applyOps,
  describeChange,
  joinBlocks,
  type BlockOp,
  parseOps,
  plainQuote,
  quoteOccurrence,
  splitBlocks
} from "./blocks.ts";
import { complete } from "./completion.ts";
import { normalizeWidgetFences, previewWithProgress } from "./widgetFences.ts";
import {
  buildFinishMessages,
  buildOpsMessages,
  buildRetryMessages,
  buildPlanMessages,
  buildSectionMessages,
  cleanSectionBody,
  factsFromOps,
  parsePlan,
  renderPlan,
  wantsWidget,
  type Plan
} from "./plan.ts";

// The document assistant. An empty document is written like Claude does: a plan first, then one section at a time. A document that
// already has content is changed with block operations, so everything the user did not ask about stays as it was.

export type StepState = "todo" | "active" | "done";
export interface Step {
  label: string;
  state: StepState;
  /** one line under the label (what a section covers) */
  detail?: string;
  /** how long into the task this step began, "0:07" (filled in by the caller) */
  time?: string;
}

export interface AssistantHooks {
  /** a live preview of the document body (the caller throttles how often it reaches the editor) */
  setDoc: (markdown: string) => void;
  setTitle: (title: string) => void;
  steps: (steps: Step[]) => void;
  reasoning: (text: string, streaming: boolean, ms: number) => void;
  /** the closing message as it is being written (shown in the chat only once the task is finished) */
  say?: (text: string) => void;
}

export interface AssistantRun {
  markdown: string;
  title: string;
  instruction: string;
  effort: ReasoningEffort;
  signal: AbortSignal;
  hooks: AssistantHooks;
}

export interface AssistantResult {
  /** the final document body */
  markdown: string;
  title: string;
  /** a one-line outcome ("Changed 2 blocks."); this is what a comment thread gets */
  summary: string;
  /** the closing message for the chat, written by the model once the task is done (Markdown); the chat falls back to `summary` */
  reply?: string;
  changed: boolean;
  stopped: boolean;
  /** questions Dara wants to put to the user as comments, each anchored to words in the document */
  questions: Array<{ quote: string; nth: number; body: string }>;
}

/** True when the user asked for something interactive; then a code block that is a runnable page is turned into a widget. */
const widgetFix = (run: AssistantRun) => (text: string) =>
  wantsWidget(run.instruction) ? normalizeWidgetFences(text) : text;
const opsFix = (run: AssistantRun, ops: BlockOp[]): BlockOp[] =>
  wantsWidget(run.instruction)
    ? ops.map((o) => (o.op === "delete" ? o : { ...o, text: normalizeWidgetFences(o.text) }))
    : ops;

const unchanged = (run: AssistantRun, summary: string, stopped: boolean): AssistantResult => ({
  markdown: run.markdown,
  title: run.title,
  summary,
  changed: false,
  stopped,
  questions: []
});

/**
 * The closing message, like Claude's: once the task is finished Dara says what is now in the document and offers a next step. It is a
 * short extra request, without thinking; if it fails or is stopped the one-line summary is used instead.
 */
async function finishReply(
  run: AssistantRun,
  steps: Step[],
  title: string,
  facts: string
): Promise<string | undefined> {
  const last = steps.length;
  run.hooks.steps([...steps, { label: "Write the summary", state: "active" }]);
  let text: string | undefined;
  try {
    const res = await complete({
      messages: buildFinishMessages(run.instruction, title, facts),
      temperature: 0.5,
      maxTokens: 320,
      signal: run.signal,
      onContent: (raw) => run.hooks.say?.(cleanReply(raw))
    });
    const cleaned = cleanReply(res.content);
    if (cleaned && !res.aborted) text = cleaned;
  } catch {
    // the one-line summary will do
  }
  const done = [...steps, { label: "Write the summary", state: "done" as const }];
  run.hooks.steps(done);
  void last;
  return text;
}

/** No fence around the message and no leading blank lines. */
function cleanReply(raw: string): string {
  return raw
    .replace(/^\s*```(?:markdown|md)?[ \t]*\n/i, "")
    .replace(/\n?```\s*$/, "")
    .trim();
}

export async function runAssistant(run: AssistantRun): Promise<AssistantResult> {
  return splitBlocks(run.markdown).length === 0 ? writeNew(run) : editBlocks(run);
}

async function editBlocks(run: AssistantRun): Promise<AssistantResult> {
  const blocks = splitBlocks(run.markdown);
  const steps: Step[] = [
    {
      label: `Read the document (${blocks.length} ${blocks.length === 1 ? "block" : "blocks"})`,
      state: "done"
    },
    { label: "Work out the changes", state: "active" }
  ];
  run.hooks.steps(steps);

  const first = buildOpsMessages(blocks, run.instruction);
  const ask = (messages: typeof first) =>
    complete({
      messages,
      temperature: 0.2,
      maxTokens: 4096,
      effort: run.effort,
      signal: run.signal,
      onReasoning: run.hooks.reasoning,
      onContent: (raw) => {
        const ops = opsFix(run, parseOps(raw));
        if (ops.length)
          run.hooks.setDoc(previewWithProgress(joinBlocks(applyOps(blocks, ops).blocks)));
      }
    });
  let res = await ask(first);
  // an answer that names blocks that do not exist is sent back once, like a refused call
  const firstTry = applyOps(blocks, parseOps(res.content));
  if (!res.aborted && firstTry.missing.length) {
    steps.push({ label: "Fix block numbers", state: "active" });
    run.hooks.steps([...steps]);
    run.hooks.setDoc(run.markdown);
    res = await ask(buildRetryMessages(first, res.content, firstTry.missing, blocks.length));
    steps[steps.length - 1] = { label: "Fix block numbers", state: "done" };
  }

  const ops = opsFix(run, parseOps(res.content));
  if (ops.length === 0) {
    run.hooks.setDoc(run.markdown);
    if (res.aborted) return unchanged(run, "Stopped. Nothing was changed.", true);
    // The model answered in words instead of operations (a question about the document, or it misunderstood): show that as the reply.
    const said = res.content.trim().slice(0, 1200);
    const out = unchanged(
      run,
      said ? said : "I did not find anything to change. Try describing the change differently.",
      false
    );
    return said ? { ...out, reply: said } : out;
  }
  const applied = applyOps(blocks, ops);
  const markdown = joinBlocks(applied.blocks);
  run.hooks.setDoc(markdown);
  steps[1] = { label: "Work out the changes", state: "done" };
  run.hooks.steps([...steps]);
  const changed = markdown !== run.markdown;
  const questions = applied.questions.map((q) => {
    const quote = plainQuote(blocks[q.n - 1]!);
    return { quote, nth: quoteOccurrence(blocks, q.n - 1, quote), body: q.text };
  });
  let summary = changed || questions.length ? describeChange(applied) : "Nothing needed to change.";
  if (applied.skipped)
    summary += ` Skipped ${applied.skipped} ${applied.skipped === 1 ? "edit" : "edits"} that named a block that does not exist.`;
  if (res.aborted)
    summary = `Stopped. ${changed ? "Kept what was done so far. " : ""}${summary}`.trim();
  const out: AssistantResult = {
    markdown,
    title: run.title,
    summary,
    changed,
    stopped: res.aborted,
    questions
  };
  if (!res.aborted && (changed || questions.length)) {
    const reply = await finishReply(run, steps, run.title, factsFromOps(blocks, ops));
    if (reply) out.reply = reply;
  }
  return out;
}

async function writeNew(run: AssistantRun): Promise<AssistantResult> {
  const steps: Step[] = [{ label: "Plan the document", state: "active" }];
  run.hooks.steps(steps);

  // 1. the plan; the outline appears in the page as it streams
  let plan: Plan = { title: "", sections: [] };
  const planRes = await complete({
    messages: buildPlanMessages(run.instruction),
    temperature: 0.4,
    maxTokens: 700,
    effort: run.effort,
    signal: run.signal,
    onReasoning: run.hooks.reasoning,
    onContent: (raw) => {
      plan = parsePlan(raw);
      if (plan.sections.length) {
        run.hooks.setDoc(renderPlan(plan, []));
        run.hooks.steps([
          { label: "Plan the document", state: "active" },
          ...plan.sections.map((s) => ({
            label: s.heading,
            detail: s.intent,
            state: "todo" as const
          }))
        ]);
      }
    }
  });
  plan = parsePlan(planRes.content);
  if (planRes.aborted) {
    run.hooks.setDoc(run.markdown);
    return unchanged(run, "Stopped before writing anything.", true);
  }

  // The model did not follow the plan format: write the whole document in one go instead.
  if (plan.sections.length === 0) {
    run.hooks.steps([{ label: "Write the document", state: "active" }]);
    const text = await runEdit({
      markdown: "",
      instruction: run.instruction,
      signal: run.signal,
      onText: run.hooks.setDoc
    });
    if (!text.trim()) {
      run.hooks.setDoc(run.markdown);
      return unchanged(
        run,
        "I could not write that. Try describing the document differently.",
        run.signal.aborted
      );
    }
    run.hooks.steps([{ label: "Write the document", state: "done" }]);
    return {
      markdown: text,
      title: run.title,
      summary: "Wrote the document.",
      changed: true,
      stopped: run.signal.aborted,
      questions: []
    };
  }

  const title = run.title.trim() ? run.title : plan.title;
  if (title !== run.title) run.hooks.setTitle(title);

  // 2. the sections, one request each
  const fix = widgetFix(run);
  const bodies: string[] = plan.sections.map(() => "");
  const stepsNow = (active: number): Step[] => [
    { label: "Plan the document", state: "done" },
    ...plan.sections.map((s, i) => ({
      label: s.heading,
      detail: s.intent,
      state: (i < active ? "done" : i === active ? "active" : "todo") as StepState
    }))
  ];
  run.hooks.setDoc(renderPlan(plan, bodies));
  let stopped = false;
  for (let i = 0; i < plan.sections.length; i++) {
    if (run.signal.aborted) {
      stopped = true;
      break;
    }
    run.hooks.steps(stepsNow(i));
    const sec = plan.sections[i]!;
    const res = await complete({
      messages: buildSectionMessages(run.instruction, plan, i, bodies),
      temperature: 0.5,
      maxTokens: 1200,
      signal: run.signal,
      onContent: (raw) => {
        bodies[i] = fix(cleanSectionBody(raw, sec.heading));
        // a widget that is still being written shows progress, not a wall of code
        run.hooks.setDoc(renderPlan(plan, bodies.map(previewWithProgress)));
      }
    });
    bodies[i] = fix(cleanSectionBody(res.content, sec.heading));
    if (res.aborted) {
      stopped = true;
      break;
    }
  }

  // sections that were never reached are dropped rather than left as placeholders
  const kept = plan.sections.map((s, i) => ({ s, body: bodies[i]!.trim() })).filter((x) => x.body);
  const markdown = kept.map((x) => `## ${x.s.heading}\n\n${x.body}`).join("\n\n");
  run.hooks.setDoc(markdown);
  run.hooks.steps(
    stopped
      ? stepsNow(kept.length).map((st, i) =>
          i > 0 && i <= kept.length ? { ...st, state: "done" as const } : st
        )
      : stepsNow(plan.sections.length)
  );
  const n = kept.length;
  const summary = stopped
    ? n
      ? `Stopped after ${n} of ${plan.sections.length} sections. Kept what was written.`
      : "Stopped before any section was written."
    : `Wrote ${title ? `"${title}"` : "the document"} in ${n} ${n === 1 ? "section" : "sections"}.`;
  const out: AssistantResult = {
    markdown,
    title: n > 0 ? title : run.title,
    summary,
    changed: n > 0,
    stopped,
    questions: []
  };
  if (!stopped && n > 0) {
    const facts = kept
      .map((x) => `- ${x.s.heading}: ${x.body.replace(/\s+/g, " ").slice(0, 140)}`)
      .join("\n");
    const reply = await finishReply(run, stepsNow(plan.sections.length), title, facts);
    if (reply) out.reply = reply;
  }
  return out;
}
