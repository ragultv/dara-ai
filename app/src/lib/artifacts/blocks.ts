// Targeted edits: the document is shown to the model as numbered blocks, and the model answers with a few operations on those numbers
// instead of the whole document. Blocks it does not mention stay exactly as they were, byte for byte.

/** Splits Markdown into top-level blocks (separated by blank lines). A fenced code block stays in one piece even if it holds blank lines. */
export function splitBlocks(markdown: string): string[] {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const blocks: string[] = [];
  let cur: string[] = [];
  let fence: string | null = null;
  const flush = () => {
    const text = cur.join("\n").replace(/\s+$/, "");
    if (text.trim()) blocks.push(text);
    cur = [];
  };
  for (const line of lines) {
    const f = /^\s{0,3}(`{3,}|~{3,})/.exec(line);
    if (f) {
      const mark = f[1]!;
      if (fence === null) fence = mark;
      else if (mark[0] === fence[0] && mark.length >= fence.length) fence = null;
    }
    if (fence === null && line.trim() === "") flush();
    else cur.push(line);
  }
  flush();
  return blocks;
}

export function joinBlocks(blocks: string[]): string {
  return blocks.join("\n\n");
}

/** The document as the model reads it: `[1] first block`, `[2] ...`. Long blocks are not shortened: the model needs the real text. */
export function numberBlocks(blocks: string[]): string {
  return blocks.map((b, i) => `[${i + 1}]\n${b}`).join("\n\n");
}

export type BlockOp =
  | { op: "replace"; n: number; text: string }
  | { op: "insert"; n: number; text: string } // after block n; 0 = at the very start
  | { op: "delete"; n: number }
  | { op: "comment"; n: number; text: string }; // a question for the user about block n; the text is left as it is

const OP_LINE = /^@(replace|insert|delete|comment)(?:\s+(after|before))?\s+\[?(\d+)\]?\s*:?\s*$/i;

/**
 * Reads the model's answer. Each operation starts on its own line:
 *   @replace 3   (the new text of block 3 follows)
 *   @insert 3    (new blocks go after block 3; @insert 0 puts them at the start)
 *   @insert before 4   (the same as @insert 3: a model copies the words of "before the last slide" more reliably than it subtracts)
 *   @delete 5
 * Text before the first operation is ignored. Works on a partial answer while it streams.
 */
export function parseOps(raw: string): BlockOp[] {
  const ops: BlockOp[] = [];
  let cur: { op: "replace" | "insert" | "delete" | "comment"; n: number; body: string[] } | null =
    null;
  let fence: string | null = null;
  const close = () => {
    if (!cur) return;
    const text = cur.body.join("\n").trim();
    if (cur.op === "delete") ops.push({ op: "delete", n: cur.n });
    else if (text) ops.push({ op: cur.op, n: cur.n, text });
    cur = null;
  };
  for (const line of raw.replace(/\r\n/g, "\n").split("\n")) {
    const f = /^\s{0,3}(`{3,}|~{3,})/.exec(line);
    if (f && cur) {
      const mark = f[1]!;
      if (fence === null) fence = mark;
      else if (mark[0] === fence[0] && mark.length >= fence.length) fence = null;
    }
    if (fence === null) {
      const m = OP_LINE.exec(line.trim());
      if (m) {
        close();
        const op = m[1]!.toLowerCase() as "replace" | "insert" | "delete" | "comment";
        // "before 4" is after 3; only an insert has a position, so the word means nothing to the other operations
        const before = op === "insert" && m[2]?.toLowerCase() === "before";
        cur = { op, n: Math.max(0, Number(m[3]) - (before ? 1 : 0)), body: [] };
        continue;
      }
      if (/^@(done|end)\s*$/i.test(line.trim())) {
        close();
        continue;
      }
    }
    if (cur) cur.body.push(line);
  }
  close();
  return ops;
}

export interface ApplyResult {
  blocks: string[];
  replaced: number;
  inserted: number;
  deleted: number;
  /** operations that named a block that does not exist (or that were repeated) */
  skipped: number;
  /** the block numbers that do not exist */
  missing: number[];
  /** questions Dara wants to ask, each about an existing block */
  questions: Array<{ n: number; text: string }>;
}

/** Applies operations to the original blocks. Numbers always refer to the original numbering. */
export function applyOps(blocks: string[], ops: BlockOp[]): ApplyResult {
  const replace = new Map<number, string>();
  const remove = new Set<number>();
  const after = new Map<number, string[]>();
  let skipped = 0;
  const missing: number[] = [];
  const questions: Array<{ n: number; text: string }> = [];
  for (const o of ops) {
    const exists = o.n >= 1 && o.n <= blocks.length;
    if (o.op === "comment") {
      if (exists) questions.push({ n: o.n, text: o.text });
      else {
        skipped++;
        missing.push(o.n);
      }
    } else if (o.op === "insert") {
      const at = Math.min(Math.max(0, o.n), blocks.length); // after the end is the end
      after.set(at, [...(after.get(at) ?? []), o.text]);
    } else if (!exists) {
      skipped++;
      missing.push(o.n);
    } else if (o.op === "replace") {
      if (remove.has(o.n)) skipped++;
      else replace.set(o.n, o.text);
    } else {
      remove.add(o.n);
      replace.delete(o.n);
    }
  }
  const out: string[] = [...(after.get(0) ?? [])];
  blocks.forEach((b, i) => {
    const n = i + 1;
    if (!remove.has(n)) out.push(replace.get(n) ?? b);
    out.push(...(after.get(n) ?? []));
  });
  return {
    blocks: out,
    replaced: replace.size,
    inserted: [...after.values()].reduce((a, v) => a + v.length, 0),
    deleted: remove.size,
    skipped,
    missing,
    questions
  };
}

/** "Changed 2 blocks, added 1, removed 1." for the chat line. */
export function describeChange(
  r: Pick<ApplyResult, "replaced" | "inserted" | "deleted"> & { questions?: unknown[] },
  noun: { one: string; many: string } = { one: "block", many: "blocks" }
): string {
  const part = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;
  const bits: string[] = [];
  if (r.replaced) bits.push(`changed ${part(r.replaced, noun.one, noun.many)}`);
  if (r.inserted) bits.push(`added ${part(r.inserted, noun.one, noun.many)}`);
  if (r.deleted) bits.push(`removed ${part(r.deleted, noun.one, noun.many)}`);
  const asked = r.questions?.length ?? 0;
  if (asked) bits.push(`asked ${part(asked, "question", "questions")} in the comments`);
  if (!bits.length) return "Nothing needed to change.";
  const s = bits.join(", ");
  return s[0]!.toUpperCase() + s.slice(1) + ".";
}

/** The plain words a block starts with (no Markdown marks), as a comment anchor. Empty when the block has no words to point at (code). */
export function plainQuote(block: string): string {
  let t = block.split("\n")[0] ?? "";
  if (/^\s*(```|~~~)/.test(t)) return "";
  if (/^\s*\|/.test(t))
    t =
      t
        .split("|")
        .map((c) => c.trim())
        .find((c) => c && !/^[-: ]+$/.test(c)) ?? "";
  t = t.replace(/^\s{0,3}(#{1,6}\s+|>\s?|[-*+]\s+(\[[ xX]\]\s+)?|\d+[.)]\s+)/, "");
  t = t
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/[*_`~]/g, "")
    .trim();
  if (t.length > 80) {
    t = t.slice(0, 80);
    const sp = t.lastIndexOf(" ");
    if (sp > 40) t = t.slice(0, sp);
  }
  return t.trim();
}

/** Which occurrence of the block's quote this is: how many times the same words appear in the blocks before it. */
export function quoteOccurrence(blocks: readonly string[], index: number, quote: string): number {
  if (!quote) return 0;
  let n = 0;
  for (let i = 0; i < index; i++) {
    const plain = blocks[i]!.replace(/[*_`~]/g, "");
    let at = plain.indexOf(quote);
    while (at !== -1) {
      n++;
      at = plain.indexOf(quote, at + quote.length);
    }
  }
  return n;
}
