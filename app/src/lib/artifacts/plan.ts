import type { ChatMessage } from "../api/types.ts";
import { cleanEditResult } from "./askDara.ts";
import { numberBlocks, type BlockOp } from "./blocks.ts";

// Dara writes a document the way Claude does: first a plan (a title and the sections), then one section at a time into the page.
// A document that already has content is edited with block operations instead (blocks.ts).

export interface PlanSection {
  heading: string;
  intent: string;
}
export interface Plan {
  title: string;
  sections: PlanSection[];
}

export const MAX_SECTIONS = 8;

/** The user asks for something interactive (a widget, calculator, chart, converter, quiz, timer...). */
export function wantsWidget(instruction: string): boolean {
  return /\b(widgets?|interactive|calculators?|converters?|simulators?|quiz(?:zes)?|timers?|sliders?|demos?|playgrounds?|visuali[sz]ers?|charts?|graphs?|diagrams?|flow ?charts?|infographics?|schematics?|mind ?maps?)\b/i.test(
    instruction
  );
}

/** The user asks for a picture of how things fit together (a diagram, flow, chart) rather than a control. */
export function wantsDiagram(instruction: string): boolean {
  return /\b(diagrams?|flow ?charts?|architecture|infographics?|schematics?|mind ?maps?|charts?|graphs?)\b/i.test(
    instruction
  );
}

/**
 * How to write a widget: a fenced code block whose language is "widget" holding one small self-contained HTML page. The app shows it live
 * in a sandboxed frame with no network, so it must not need anything else.
 */
export const WIDGET_GUIDE = [
  "For something interactive, write a widget: a fenced code block whose language is widget, like this:",
  "```widget",
  '<div id="app"><label for="a">Amount</label><input id="a" type="number" value="10"><button id="go">Calculate</button><p id="out"></p></div>',
  "<style>#app{max-width:340px}#out{font-size:1.2em;font-weight:600}</style>",
  "<script>document.getElementById('go').onclick=()=>{document.getElementById('out').textContent=Number(document.getElementById('a').value)*2}</script>",
  "```",
  "Rules for a widget: one block, under 70 lines, plain HTML, CSS in a <style> element and JavaScript in a <script> element, no external files, no frameworks, no network, no alert() or prompt().",
  "Colours ONLY from the page's variables, so it matches light and dark mode: var(--text), var(--muted), var(--bg), var(--surface), var(--border), var(--primary). Never write a colour like #fff or green. Buttons, inputs, labels, tables and links are already styled: do not restyle them.",
  "It must work on its own: connect every button and input with ids and event handlers in the script, and show results in the page."
].join("\n");

/** How to draw a diagram or chart inside a widget: inline SVG in the page's own colours. */
export const DIAGRAM_GUIDE = [
  "For a diagram, flow or chart, draw it as inline SVG inside the widget, with a title line above it that states the finding:",
  "```widget",
  '<svg viewBox="0 0 640 140" width="100%"><defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--muted)"/></marker></defs>',
  '<rect x="20" y="30" width="220" height="70" rx="10" fill="var(--surface)" stroke="var(--border)"/><text x="130" y="70" text-anchor="middle">Request</text>',
  '<line x1="240" y1="65" x2="390" y2="65" stroke="var(--muted)" marker-end="url(#ar)"/>',
  '<rect x="400" y="30" width="220" height="70" rx="10" fill="var(--surface)" stroke="var(--primary)" stroke-width="2"/><text x="510" y="70" text-anchor="middle">Engine</text></svg>',
  "```",
  "Boxes use fill var(--surface) and stroke var(--border); text uses the default fill (var(--text)) or var(--muted) for small notes; use var(--primary) as the ONE accent, on the part the title is about.",
  "Keep every label short enough to fit inside its box, leave gaps between boxes, never let text overlap a line or another label, and draw arrows with marker-end."
].join("\n");

/** The guide for what the user asked: a widget always, plus how to draw when it is a diagram. */
export function widgetGuide(instruction: string): string {
  return wantsDiagram(instruction) ? `${WIDGET_GUIDE}\n\n${DIAGRAM_GUIDE}` : WIDGET_GUIDE;
}

export const PLAN_SYSTEM_PROMPT = [
  "You are Dara, the writing assistant inside a document editor.",
  "Plan a document for the user's request. Reply with the plan only, in exactly this shape:",
  "# <document title>",
  "## <section heading> :: <one sentence on what the section covers>",
  `Use 3 to ${MAX_SECTIONS} sections, one per line, short headings, no numbering. No introduction, no notes, no other text.`
].join("\n");

export function buildPlanMessages(instruction: string): ChatMessage[] {
  const widget = wantsWidget(instruction)
    ? '\nThe user wants something interactive: make exactly one section whose heading names it (for example "Calculator widget").'
    : "";
  return [
    { role: "system", content: PLAN_SYSTEM_PROMPT + widget },
    { role: "user", content: `Request: ${instruction.trim()}` }
  ];
}

/** Reads the plan; tolerant of bullets, numbering, a missing "::" and a missing title. Works on partial text. */
export function parsePlan(raw: string): Plan {
  let title = "";
  const sections: PlanSection[] = [];
  for (const line of cleanEditResult(raw).split("\n")) {
    const t = line.trim();
    if (!t) continue;
    const h1 = /^#\s+(.+)$/.exec(t);
    if (h1 && !title && sections.length === 0) {
      title = stripMd(h1[1]!);
      continue;
    }
    const item = /^(?:#{1,6}\s+|[-*]\s+|\d+[.)]\s+)(.+)$/.exec(t);
    if (!item) continue;
    const [headingRaw, ...rest] = item[1]!.replace(/(\*\*|__):/, "$1 ::").split(/\s*(?:::|—|–)\s*/);
    const heading = stripMd(headingRaw ?? "");
    if (!heading) continue;
    sections.push({ heading, intent: stripMd(rest.join(" — ")) });
    if (sections.length >= MAX_SECTIONS) break;
  }
  return { title, sections };
}

function stripMd(s: string): string {
  return s
    .replace(/[*_`]/g, "")
    .replace(/^["']|["']$/g, "")
    .trim();
}

export const SECTION_SYSTEM_PROMPT = [
  "You are Dara, the writing assistant inside a document editor.",
  "Write the body of ONE section of a document, in Markdown.",
  "Do not write the section heading or the document title. Do not greet, explain or add notes.",
  "Be concrete and specific. Use short paragraphs, - bullet lists, 1. numbered lists or a | table | where they help.",
  "Do not repeat what other sections cover. Do not wrap the answer in a code fence unless it is code."
].join(" ");

export function buildSectionMessages(
  request: string,
  plan: Plan,
  index: number,
  written: string[]
): ChatMessage[] {
  const outline = plan.sections.map(
    (s, i) => `${i + 1}. ${s.heading}${s.intent ? ` - ${s.intent}` : ""}`
  );
  const sec = plan.sections[index]!;
  const before = written
    .map((body, i) => (body ? `## ${plan.sections[i]!.heading}\n${body}` : ""))
    .filter(Boolean)
    .join("\n\n")
    .slice(-1800);
  const isWidgetSection =
    wantsWidget(request) &&
    /widget|calculator|converter|interactive|demo|quiz|timer|simulator/i.test(sec.heading);
  const system = isWidgetSection
    ? `${SECTION_SYSTEM_PROMPT}\n\nThis section is the interactive widget: write one short sentence introducing it, then the widget.\n${widgetGuide(request)}`
    : SECTION_SYSTEM_PROMPT;
  return [
    { role: "system", content: system },
    {
      role: "user",
      content: [
        `Document: ${plan.title || "(untitled)"}`,
        `User request: ${request.trim()}`,
        `Outline:\n${outline.join("\n")}`,
        before ? `Already written (for continuity, do not repeat):\n${before}` : "",
        `Now write section ${index + 1}: "${sec.heading}"${sec.intent ? ` (${sec.intent})` : ""}.`
      ]
        .filter(Boolean)
        .join("\n\n")
    }
  ];
}

/** A section body from the model: no fence around it and no repeated heading line. */
export function cleanSectionBody(raw: string, heading: string): string {
  let t = cleanEditResult(raw).replace(/^\s+/, "");
  const first = /^(#{1,6})\s+(.+)\n?/.exec(t);
  if (first && stripMd(first[2]!).toLowerCase() === heading.toLowerCase())
    t = t.slice(first[0].length);
  return t.replace(/^\n+/, "");
}

export const PENDING_TEXT = "*Dara is writing this section...*";

/** The page while the sections are being written. Sections not started yet show a placeholder. */
export function renderPlan(plan: Plan, bodies: string[]): string {
  return plan.sections
    .map((s, i) => `## ${s.heading}\n\n${bodies[i]?.trim() ? bodies[i] : PENDING_TEXT}`)
    .join("\n\n");
}

export const OPS_SYSTEM_PROMPT = [
  "You are Dara, the writing assistant inside a document editor.",
  "The document is shown as numbered blocks like [1], [2]. Change it with operations, one per line start, and reply with nothing else:",
  "@replace N   - the new Markdown for block N follows on the next lines",
  "@insert N    - new Markdown blocks follow, placed after block N (@insert 0 puts them at the start)",
  "@delete N    - remove block N",
  "@comment N   - ask the user a question about block N; the question follows on the next lines. Use it ONLY when you truly need a decision from the user: do not guess, ask, and leave that text as it is.",
  "Only touch the blocks the instruction needs. Blocks you do not mention stay exactly as they are, so never repeat them.",
  "Write the new text without the [N] marker. Use plain Markdown: # headings, **bold**, - lists, | tables |, > quotes, fenced code.",
  "A list or a table is ONE block: replace it as a whole. No greeting, no explanation, no code fence around the answer."
].join("\n");

/**
 * Editing a deck. The numbered units are whole SLIDES, not paragraphs: a small model counts slides reliably, and it does not have to
 * keep or place the --- lines (the deck is written back with them). The same operations as a document, so the same parser applies them.
 */
export const DECK_OPS_SYSTEM_PROMPT = [
  "You are Dara, the assistant inside a slide editor.",
  "The slide deck is shown as numbered slides like [1], [2]. Change it with operations, one per line start, and reply with nothing else:",
  "@replace N   - the complete new Markdown of slide N follows on the next lines",
  "@insert after N   - the Markdown of a NEW slide follows on the next lines, placed after slide N",
  "@insert before N  - the same, placed before slide N (@insert before 1 puts it first)",
  "@delete N    - remove slide N",
  "Only touch the slides the instruction needs. Slides you do not mention stay exactly as they are, so never repeat them.",
  "A slide is a ## heading and 3 to 5 short - bullets (each under 12 words). The first slide is the title slide: a # heading and one short line.",
  "Write the slide without the [N] marker and without --- lines. A replaced slide must contain the WHOLE slide, including the lines that do not change (the subtitle of a title slide, the other bullets).",
  "To change the deck's title, replace slide 1 (keep its subtitle line). To reorder, delete and insert. Each slide is written once: never insert a slide you also replaced.",
  "No greeting, no explanation, no code fence.",
  "Example. Slides: [1] a # Plan heading and the line 'Our year'; [2] a ## Goals slide with two bullets. Instruction: rename the deck to Roadmap and add a closing slide. Answer:",
  "@replace 1",
  "# Roadmap",
  "",
  "Our year",
  "@insert after 2",
  "## Thanks",
  "",
  "- Questions?"
].join("\n");

export function buildOpsMessages(
  blocks: string[],
  instruction: string,
  kind: "doc" | "slides" = "doc"
): ChatMessage[] {
  const widget =
    kind === "doc" && wantsWidget(instruction)
      ? `\n\nThe user wants something interactive, so add it with @insert (a widget is ONE block).\n${widgetGuide(instruction)}`
      : "";
  if (kind === "slides")
    return [
      { role: "system", content: DECK_OPS_SYSTEM_PROMPT },
      {
        role: "user",
        content: `Slides:\n${numberBlocks(blocks)}\n\nInstruction: ${instruction.trim()}`
      }
    ];
  return [
    { role: "system", content: OPS_SYSTEM_PROMPT + widget },
    {
      role: "user",
      content: `Document:\n${numberBlocks(blocks)}\n\nInstruction: ${instruction.trim()}`
    }
  ];
}

/**
 * The follow-up when an answer named blocks that do not exist. Claude Docs refuses such a call and the model resends it corrected; Dara
 * does the same once, quoting its own answer back with the real range.
 */
export function buildRetryMessages(
  first: ChatMessage[],
  answer: string,
  missing: number[],
  blockCount: number,
  noun: { one: string; many: string } = { one: "block", many: "blocks" }
): ChatMessage[] {
  const list = [...new Set(missing)].join(", ");
  return [
    ...first,
    { role: "assistant", content: answer },
    {
      role: "user",
      content: `That was refused and nothing was applied: ${noun.one} ${list} ${missing.length === 1 ? "does" : "do"} not exist. The ${noun.one === "slide" ? "deck" : "document"} has ${blockCount} ${blockCount === 1 ? noun.one : noun.many}, numbered 1 to ${blockCount}. Resend all the operations, corrected, in the same format.`
    }
  ];
}

export const FINISH_SYSTEM_PROMPT = [
  "You are Dara, the writing assistant in a document editor. You have just finished the user's task.",
  "Write the closing message to the user in plain, friendly language: first one or two sentences saying what is now in the document, naming it;",
  "then a short bullet list (at most five lines, each starting with '- ') of what it contains or what changed;",
  "then one short sentence offering a next step.",
  "Use only the facts you are given. No headings, no tables, no code, and never mention tools, blocks or operations."
].join(" ");

export function buildFinishMessages(
  instruction: string,
  title: string,
  facts: string,
  kind: "doc" | "slides" = "doc"
): ChatMessage[] {
  const system =
    kind === "slides"
      ? FINISH_SYSTEM_PROMPT.replace("document editor", "slide editor")
          .replace("is now in the document", "is now in the deck")
          .replace("tools, blocks or operations", "tools, blocks, operations or the --- marks")
      : FINISH_SYSTEM_PROMPT;
  return [
    { role: "system", content: system },
    {
      role: "user",
      content: `Task: ${instruction.trim().slice(0, 500)}\n${kind === "slides" ? "Slide deck" : "Document"}: ${title || "(untitled)"}\nWhat was done:\n${facts.slice(0, 2500)}`
    }
  ];
}

/** What an edit did, as plain facts for the closing message (the words, not the block numbers). */
export function factsFromOps(blocks: readonly string[], ops: readonly BlockOp[]): string {
  const short = (t: string) => t.replace(/\s+/g, " ").trim().slice(0, 160);
  const lines: string[] = [];
  for (const o of ops.slice(0, 8)) {
    if (o.op === "replace")
      lines.push(`- Changed "${short(blocks[o.n - 1] ?? "")}" to: ${short(o.text)}`);
    else if (o.op === "insert") lines.push(`- Added: ${short(o.text)}`);
    else if (o.op === "delete") lines.push(`- Removed: ${short(blocks[o.n - 1] ?? "")}`);
    else lines.push(`- Asked the user in a comment: ${short(o.text)}`);
  }
  return lines.join("\n");
}

export const DECK_SYSTEM_PROMPT = [
  "You are Dara, the assistant inside a slide editor. Write the slide deck the user asks for, as Markdown.",
  "Rules:",
  "- Slide 1 is the title slide: a # heading and one short line under it.",
  "- Then 5 to 9 slides (fewer if the user asks for fewer). Every slide starts with a ## heading.",
  "- Under the heading: 3 to 5 short bullet lines starting with '- ', each under 12 words. No paragraphs, no tables, no images, no code unless asked.",
  "- Put a line with only --- between slides, with a blank line before and after it.",
  "- End with a short closing slide (for example next steps or questions).",
  "Reply with the deck only: no greeting, no explanation, no code fence.",
  "Example of the shape:",
  "# Quarterly review",
  "",
  "How the team did and what comes next",
  "",
  "---",
  "",
  "## Highlights",
  "",
  "- Shipped the new onboarding",
  "- Support replies twice as fast",
  "",
  "---",
  "",
  "## Next steps",
  "",
  "- Hire two engineers",
  "- Start the pilot in March"
].join("\n");

export function buildDeckMessages(instruction: string): ChatMessage[] {
  return [
    { role: "system", content: DECK_SYSTEM_PROMPT },
    { role: "user", content: `Write this deck: ${instruction.trim()}` }
  ];
}
