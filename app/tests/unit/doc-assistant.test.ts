import { describe, expect, it } from "vitest";
import {
  applyOps,
  describeChange,
  joinBlocks,
  numberBlocks,
  parseOps,
  splitBlocks
} from "../../src/lib/artifacts/blocks.ts";
import {
  buildSectionMessages,
  cleanSectionBody,
  parsePlan,
  PENDING_TEXT,
  renderPlan
} from "../../src/lib/artifacts/plan.ts";

describe("splitBlocks", () => {
  it("splits on blank lines and keeps a fenced block whole", () => {
    const md = "# T\n\npara one\nstill one\n\n```js\na\n\nb\n```\n\n- x\n- y";
    expect(splitBlocks(md)).toEqual([
      "# T",
      "para one\nstill one",
      "```js\na\n\nb\n```",
      "- x\n- y"
    ]);
  });
  it("round-trips through joinBlocks", () => {
    const md = "# T\n\nbody\n\n| a | b |\n| - | - |\n| 1 | 2 |";
    expect(joinBlocks(splitBlocks(md))).toBe(md);
  });
  it("ignores an empty document", () => {
    expect(splitBlocks("  \n\n")).toEqual([]);
  });
  it("numbers blocks from 1", () => {
    expect(numberBlocks(["a", "b"])).toBe("[1]\na\n\n[2]\nb");
  });
});

describe("parseOps", () => {
  it("reads replace, insert and delete", () => {
    const ops = parseOps("@replace 2\nnew two\n\n@insert 3\n## Added\n\ntext\n@delete 5\n@done");
    expect(ops).toEqual([
      { op: "replace", n: 2, text: "new two" },
      { op: "insert", n: 3, text: "## Added\n\ntext" },
      { op: "delete", n: 5 }
    ]);
  });
  it("tolerates 'insert after 3', brackets and a preamble", () => {
    expect(parseOps("Sure!\n@insert after [3]:\nhello")).toEqual([
      { op: "insert", n: 3, text: "hello" }
    ]);
  });
  it("does not read an @ line inside a code fence as an operation", () => {
    const ops = parseOps("@replace 1\n```\n@delete 2\n```");
    expect(ops).toEqual([{ op: "replace", n: 1, text: "```\n@delete 2\n```" }]);
  });
  it("works on a partial answer and drops an empty replace", () => {
    expect(parseOps("@replace 1\nhalf a sen")).toEqual([
      { op: "replace", n: 1, text: "half a sen" }
    ]);
    expect(parseOps("@replace 1\n")).toEqual([]);
    expect(parseOps("no operations here")).toEqual([]);
  });
});

describe("applyOps", () => {
  const blocks = ["# T", "one", "two", "three"];
  it("changes only what is named and uses the original numbering", () => {
    const r = applyOps(blocks, [
      { op: "replace", n: 2, text: "ONE" },
      { op: "delete", n: 3 },
      { op: "insert", n: 4, text: "four" },
      { op: "insert", n: 0, text: "first" }
    ]);
    expect(r.blocks).toEqual(["first", "# T", "ONE", "three", "four"]);
    expect([r.replaced, r.inserted, r.deleted, r.skipped]).toEqual([1, 2, 1, 0]);
  });
  it("skips a missing block, treats insert past the end as append", () => {
    const r = applyOps(blocks, [
      { op: "replace", n: 9, text: "x" },
      { op: "delete", n: 0 },
      { op: "insert", n: 99, text: "end" }
    ]);
    expect(r.blocks).toEqual([...blocks, "end"]);
    expect(r.skipped).toBe(2);
  });
  it("a delete beats a replace of the same block", () => {
    const r = applyOps(blocks, [
      { op: "replace", n: 2, text: "x" },
      { op: "delete", n: 2 }
    ]);
    expect(r.blocks).toEqual(["# T", "two", "three"]);
  });
  it("leaves untouched blocks byte for byte", () => {
    const md = "# T\n\n- a\n- b\n\n| a |\n| - |\n| 1 |";
    const b = splitBlocks(md);
    expect(joinBlocks(applyOps(b, []).blocks)).toBe(md);
  });
});

describe("describeChange", () => {
  it("reads like a chat line", () => {
    expect(describeChange({ replaced: 2, inserted: 1, deleted: 0 })).toBe(
      "Changed 2 blocks, added 1 block."
    );
    expect(describeChange({ replaced: 0, inserted: 0, deleted: 1 })).toBe("Removed 1 block.");
    expect(describeChange({ replaced: 0, inserted: 0, deleted: 0 })).toBe(
      "Nothing needed to change."
    );
  });
});

describe("parsePlan", () => {
  it("reads a title and sections with intents", () => {
    const p = parsePlan(
      "# Project Brief\n## Goals :: what we want\n## Scope :: in and out\n## Risks"
    );
    expect(p.title).toBe("Project Brief");
    expect(p.sections).toEqual([
      { heading: "Goals", intent: "what we want" },
      { heading: "Scope", intent: "in and out" },
      { heading: "Risks", intent: "" }
    ]);
  });
  it("tolerates bullets, numbering, a fence and no title", () => {
    const p = parsePlan("```markdown\n1. **Intro**: x\n- Plan — the steps\n```");
    expect(p.title).toBe("");
    expect(p.sections.map((s) => s.heading)).toEqual(["Intro", "Plan"]);
  });
  it("caps the number of sections and works on partial text", () => {
    const many = Array.from({ length: 20 }, (_, i) => `## S${i}`).join("\n");
    expect(parsePlan(many).sections).toHaveLength(8);
    expect(parsePlan("# Tit").title).toBe("Tit");
  });
});

describe("sections", () => {
  const plan = {
    title: "Doc",
    sections: [
      { heading: "A", intent: "first" },
      { heading: "B", intent: "" }
    ]
  };
  it("shows a placeholder for sections not written yet", () => {
    expect(renderPlan(plan, ["body a", ""])).toBe(`## A\n\nbody a\n\n## B\n\n${PENDING_TEXT}`);
  });
  it("drops a repeated heading and a fence around a section body", () => {
    expect(cleanSectionBody("## A\nhello", "A")).toBe("hello");
    expect(cleanSectionBody("```markdown\ntext\n```", "A")).toBe("text");
    expect(cleanSectionBody("## Other\nhello", "A")).toBe("## Other\nhello");
  });
  it("tells the model which section to write and what is already written", () => {
    const m = buildSectionMessages("make a doc", plan, 1, ["body a", ""]);
    const text = String(m[1]!.content);
    expect(text).toContain('write section 2: "B"');
    expect(text).toContain("body a");
    expect(text).toContain("1. A - first");
  });
});
