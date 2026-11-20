import type { ReasoningEffort } from "../chat/effort.ts";
import { runEdit } from "./askDara.ts";
import { applyOps, describeChange, joinBlocks, parseOps, splitBlocks } from "./blocks.ts";
import { complete } from "./completion.ts";
import {
  buildOpsMessages,
  buildPlanMessages,
  buildSectionMessages,
  cleanSectionBody,
  parsePlan,
  renderPlan,
  type Plan
} from "./plan.ts";

// The document assistant. An empty document is written like Claude does: a plan first, then one section at a time. A document that
// already has content is changed with block operations, so everything the user did not ask about stays as it was.

export type StepState = "todo" | "active" | "done";
export interface Step {
  label: string;
  state: StepState;
}

export interface AssistantHooks {
  /** a live preview of the document body (the caller throttles how often it reaches the editor) */
  setDoc: (markdown: string) => void;
  setTitle: (title: string) => void;
  steps: (steps: Step[]) => void;
  reasoning: (text: string, streaming: boolean, ms: number) => void;
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
  /** what Dara tells the user in the chat */
  summary: string;
  changed: boolean;
  stopped: boolean;
}

const unchanged = (run: AssistantRun, summary: string, stopped: boolean): AssistantResult => ({
  markdown: run.markdown,
  title: run.title,
  summary,
  changed: false,
  stopped
});

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

  const res = await complete({
    messages: buildOpsMessages(blocks, run.instruction),
    temperature: 0.2,
    maxTokens: 4096,
    effort: run.effort,
    signal: run.signal,
    onReasoning: run.hooks.reasoning,
    onContent: (raw) => {
      const ops = parseOps(raw);
      if (ops.length) run.hooks.setDoc(joinBlocks(applyOps(blocks, ops).blocks));
    }
  });

  const ops = parseOps(res.content);
  if (ops.length === 0) {
    run.hooks.setDoc(run.markdown);
    if (res.aborted) return unchanged(run, "Stopped. Nothing was changed.", true);
    // The model answered in words instead of operations (a question about the document, or it misunderstood): show that as the reply.
    const said = res.content.trim().slice(0, 600);
    return unchanged(
      run,
      said ? said : "I did not find anything to change. Try describing the change differently.",
      false
    );
  }
  const applied = applyOps(blocks, ops);
  const markdown = joinBlocks(applied.blocks);
  run.hooks.setDoc(markdown);
  steps[1] = { label: "Work out the changes", state: "done" };
  run.hooks.steps([...steps]);
  const changed = markdown !== run.markdown;
  let summary = changed ? describeChange(applied) : "Nothing needed to change.";
  if (applied.skipped)
    summary += ` Skipped ${applied.skipped} ${applied.skipped === 1 ? "edit" : "edits"} that named a block that does not exist.`;
  if (res.aborted)
    summary = `Stopped. ${changed ? "Kept what was done so far. " : ""}${summary}`.trim();
  return { markdown, title: run.title, summary, changed, stopped: res.aborted };
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
          ...plan.sections.map((s) => ({ label: s.heading, state: "todo" as const }))
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
      stopped: run.signal.aborted
    };
  }

  const title = run.title.trim() ? run.title : plan.title;
  if (title !== run.title) run.hooks.setTitle(title);

  // 2. the sections, one request each
  const bodies: string[] = plan.sections.map(() => "");
  const stepsNow = (active: number): Step[] => [
    { label: "Plan the document", state: "done" },
    ...plan.sections.map((s, i) => ({
      label: s.heading,
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
        bodies[i] = cleanSectionBody(raw, sec.heading);
        run.hooks.setDoc(renderPlan(plan, bodies));
      }
    });
    bodies[i] = cleanSectionBody(res.content, sec.heading);
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
  return { markdown, title: n > 0 ? title : run.title, summary, changed: n > 0, stopped };
}
