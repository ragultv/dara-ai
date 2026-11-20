import type { ChatMessage } from "../api/types.ts";
import { cleanEditResult } from "./askDara.ts";
import { numberBlocks } from "./blocks.ts";

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

export const PLAN_SYSTEM_PROMPT = [
  "You are Dara, the writing assistant inside a document editor.",
  "Plan a document for the user's request. Reply with the plan only, in exactly this shape:",
  "# <document title>",
  "## <section heading> :: <one sentence on what the section covers>",
  `Use 3 to ${MAX_SECTIONS} sections, one per line, short headings, no numbering. No introduction, no notes, no other text.`
].join("\n");

export function buildPlanMessages(instruction: string): ChatMessage[] {
  return [
    { role: "system", content: PLAN_SYSTEM_PROMPT },
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
  return [
    { role: "system", content: SECTION_SYSTEM_PROMPT },
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
  "Only touch the blocks the instruction needs. Blocks you do not mention stay exactly as they are, so never repeat them.",
  "Write the new text without the [N] marker. Use plain Markdown: # headings, **bold**, - lists, | tables |, > quotes, fenced code.",
  "A list or a table is ONE block: replace it as a whole. No greeting, no explanation, no code fence around the answer."
].join("\n");

export function buildOpsMessages(blocks: string[], instruction: string): ChatMessage[] {
  return [
    { role: "system", content: OPS_SYSTEM_PROMPT },
    {
      role: "user",
      content: `Document:\n${numberBlocks(blocks)}\n\nInstruction: ${instruction.trim()}`
    }
  ];
}
