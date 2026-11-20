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
  | { op: "delete"; n: number };

const OP_LINE = /^@(replace|insert|delete)(?:\s+after)?\s+\[?(\d+)\]?\s*:?\s*$/i;

/**
 * Reads the model's answer. Each operation starts on its own line:
 *   @replace 3   (the new text of block 3 follows)
 *   @insert 3    (new blocks go after block 3; @insert 0 puts them at the start)
 *   @delete 5
 * Text before the first operation is ignored. Works on a partial answer while it streams.
 */
export function parseOps(raw: string): BlockOp[] {
  const ops: BlockOp[] = [];
  let cur: { op: "replace" | "insert" | "delete"; n: number; body: string[] } | null = null;
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
        cur = {
          op: m[1]!.toLowerCase() as "replace" | "insert" | "delete",
          n: Number(m[2]),
          body: []
        };
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
}

/** Applies operations to the original blocks. Numbers always refer to the original numbering. */
export function applyOps(blocks: string[], ops: BlockOp[]): ApplyResult {
  const replace = new Map<number, string>();
  const remove = new Set<number>();
  const after = new Map<number, string[]>();
  let skipped = 0;
  const missing: number[] = [];
  for (const o of ops) {
    const exists = o.n >= 1 && o.n <= blocks.length;
    if (o.op === "insert") {
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
    missing
  };
}

/** "Changed 2 blocks, added 1, removed 1." for the chat line. */
export function describeChange(r: Pick<ApplyResult, "replaced" | "inserted" | "deleted">): string {
  const part = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;
  const bits: string[] = [];
  if (r.replaced) bits.push(`changed ${part(r.replaced, "block", "blocks")}`);
  if (r.inserted) bits.push(`added ${part(r.inserted, "block", "blocks")}`);
  if (r.deleted) bits.push(`removed ${part(r.deleted, "block", "blocks")}`);
  if (!bits.length) return "Nothing needed to change.";
  const s = bits.join(", ");
  return s[0]!.toUpperCase() + s.slice(1) + ".";
}
