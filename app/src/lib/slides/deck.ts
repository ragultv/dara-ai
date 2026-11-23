// A slide deck is ONE Markdown text, stored exactly like a document (artifacts.kind = "slides"):
//
//   [//]: # (theme: midnight)        optional first line: the look of the whole deck (a Markdown comment, invisible anywhere else)
//
//   # Title slide
//
//   ---
//
//   ## A slide
//   - a point
//
// A line with only `---` between blank lines ends a slide. Markdown is what the local model reads and writes best, and it keeps history,
// chat, export and the block edits of the document assistant working unchanged. Pure functions only (tests/unit/slides-deck.test.ts).

export const THEMES = [
  { id: "paper", label: "Paper" },
  { id: "midnight", label: "Midnight" },
  { id: "ocean", label: "Ocean" },
  { id: "sunrise", label: "Sunrise" }
] as const;
export type ThemeId = (typeof THEMES)[number]["id"];
export const DEFAULT_THEME: ThemeId = "paper";

export function isTheme(v: string): v is ThemeId {
  return THEMES.some((t) => t.id === v);
}

export interface Deck {
  theme: ThemeId;
  /** the Markdown of each slide, trimmed; never empty (a new deck has one blank slide) */
  slides: string[];
}

const THEME_LINE = /^\[\/\/\]: # \(theme: ([a-z-]+)\)\s*$/;
const FENCE = /^\s{0,3}(`{3,}|~{3,})/;
const RULE = /^\s{0,3}-{3,}\s*$/;

/** Reads a deck. An unknown theme name falls back to the default; a deck with no text is one blank slide. */
export function parseDeck(markdown: string): Deck {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  let theme: ThemeId = DEFAULT_THEME;
  let start = 0;
  while (start < lines.length && lines[start]!.trim() === "") start++;
  const m = THEME_LINE.exec(lines[start] ?? "");
  if (m) {
    if (isTheme(m[1]!)) theme = m[1];
    start++;
  }
  const slides: string[] = [];
  let cur: string[] = [];
  let fence: string | null = null;
  const flush = () => {
    slides.push(cur.join("\n").trim());
    cur = [];
  };
  for (let i = start; i < lines.length; i++) {
    const line = lines[i]!;
    const f = FENCE.exec(line);
    if (f) {
      const mark = f[1]!;
      if (fence === null) fence = mark;
      else if (mark[0] === fence[0] && mark.length >= fence.length) fence = null;
    }
    // `---` right under a line of text is a heading underline (setext), not the end of a slide
    const after = i === start || lines[i - 1]!.trim() === "";
    if (fence === null && RULE.test(line) && after) flush();
    else cur.push(line);
  }
  flush();
  return { theme, slides: slides.length > 0 ? slides : [""] };
}

/** The Markdown to store. A deck of one blank slide with the default look is the empty string, like an empty document. */
export function serializeDeck(deck: Deck): string {
  const body = deck.slides.map((s) => s.trim()).join("\n\n---\n\n");
  const blank = deck.slides.length === 1 && body === "";
  const head = deck.theme === DEFAULT_THEME ? "" : `[//]: # (theme: ${deck.theme})`;
  if (blank) return head;
  return head ? `${head}\n\n${body}` : body;
}

/** Keeps `index` inside the deck. */
export function clampIndex(deck: Deck, index: number): number {
  return Math.min(Math.max(0, index), deck.slides.length - 1);
}

export function replaceSlide(deck: Deck, index: number, markdown: string): Deck {
  if (index < 0 || index >= deck.slides.length) return deck;
  return { ...deck, slides: deck.slides.map((s, i) => (i === index ? markdown.trim() : s)) };
}

/** A blank slide after `index` (at the end when `index` is past it). */
export function addSlide(deck: Deck, index: number): Deck {
  const at = Math.min(Math.max(index + 1, 0), deck.slides.length);
  return { ...deck, slides: [...deck.slides.slice(0, at), "", ...deck.slides.slice(at)] };
}

export function duplicateSlide(deck: Deck, index: number): Deck {
  if (index < 0 || index >= deck.slides.length) return deck;
  return {
    ...deck,
    slides: [
      ...deck.slides.slice(0, index + 1),
      deck.slides[index]!,
      ...deck.slides.slice(index + 1)
    ]
  };
}

/** Removes a slide; the last one left is emptied instead (a deck always has a slide). */
export function deleteSlide(deck: Deck, index: number): Deck {
  if (index < 0 || index >= deck.slides.length) return deck;
  if (deck.slides.length === 1) return { ...deck, slides: [""] };
  return { ...deck, slides: deck.slides.filter((_, i) => i !== index) };
}

/** Moves the slide at `from` so it ends up at position `to`. */
export function moveSlide(deck: Deck, from: number, to: number): Deck {
  const n = deck.slides.length;
  if (from < 0 || from >= n || to < 0 || to >= n || from === to) return deck;
  const slides = [...deck.slides];
  const [s] = slides.splice(from, 1);
  slides.splice(to, 0, s!);
  return { ...deck, slides };
}

export function isBlank(slide: string): boolean {
  return slide.trim() === "";
}

/** The words of a Markdown line without its marks: `## **Plan**` becomes `Plan`. */
function plain(line: string): string {
  return line
    .replace(/^\s{0,3}#{1,6}\s+/, "")
    .replace(/^\s*(?:[-*+]|\d+[.)])\s+(?:\[[ xX]\]\s+)?/, "")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, "")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/[*_`~]/g, "")
    .trim();
}

/** What a slide is called in the list: its first heading, else its first line of text. */
export function slideTitle(slide: string): string {
  const lines = slide.split("\n");
  const heading = lines.find((l) => /^\s{0,3}#{1,6}\s+\S/.test(l));
  const text = plain(heading ?? lines.find((l) => plain(l) !== "") ?? "");
  return text.length > 60 ? `${text.slice(0, 59).trimEnd()}…` : text;
}

export type Layout = "title" | "section" | "content";

/**
 * How a slide is laid out, from what it holds: a lone `#` heading with at most one short line under it is a title slide, a lone `##`
 * heading is a section break (both centred and large); everything else is a content slide, written from the top left.
 */
export function slideLayout(slide: string): Layout {
  const blocks = slide
    .split(/\n\s*\n/)
    .map((b) => b.trim())
    .filter(Boolean);
  if (blocks.length === 0) return "content";
  const first = blocks[0]!;
  const isHeading = /^#{1,6}\s+\S/.test(first) && !first.includes("\n");
  if (!isHeading) return "content";
  const level = /^(#+)/.exec(first)![1]!.length;
  const rest = blocks.slice(1);
  const plainLine = (b: string) => !b.includes("\n") && !/^(?:[-*+]|\d+[.)]|>|\||```|!\[)/.test(b);
  if (level === 1 && rest.length <= 1 && rest.every(plainLine)) return "title";
  if (level === 2 && rest.length === 0) return "section";
  return "content";
}

/** The deck as it is given to the model when it is empty of everything but its look: nothing to read. */
export function deckIsEmpty(markdown: string): boolean {
  const d = parseDeck(markdown);
  return d.slides.every(isBlank);
}

/** A slide's Markdown as the first slide of a deck summary (the gallery card). */
export function firstSlide(markdown: string): { theme: ThemeId; slide: string } {
  const d = parseDeck(markdown);
  return { theme: d.theme, slide: d.slides[0] ?? "" };
}

const squash = (t: string) => t.replace(/\s+/g, " ").trim().toLowerCase();

/**
 * Drops the operations of a deck edit that only repeat something. A small model that has said what it was asked to say sometimes keeps
 * going and writes the rest of the deck back as new slides; an insert whose text holds a whole slide that is already there (or the text of
 * an operation before it) adds nothing the person asked for. `ops` are the parsed operations, `slides` the deck as the model was shown it.
 */
export function dropRepeats<T extends { op: string; text?: string }>(
  ops: T[],
  slides: readonly string[]
): T[] {
  const existing = slides.map(squash).filter((s) => s.length >= 12);
  const seen = new Set<string>();
  return ops.filter((o) => {
    if (o.op === "delete" || o.text === undefined) return true;
    const t = squash(o.text);
    if (o.op === "insert") {
      if (existing.some((s) => t.includes(s))) return false;
      if (seen.has(t)) return false;
    }
    seen.add(t);
    return true;
  });
}
