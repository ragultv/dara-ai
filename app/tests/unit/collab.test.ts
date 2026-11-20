import { describe, expect, it } from "vitest";
import { Editor } from "@tiptap/react";
import StarterKit from "@tiptap/starter-kit";
import {
  applyOps,
  describeChange,
  parseOps,
  plainQuote,
  quoteOccurrence
} from "../../src/lib/artifacts/blocks.ts";
import {
  awaitsDara,
  buildThreads,
  commentInstruction,
  mentionsDara,
  threadForPrompt
} from "../../src/lib/artifacts/threads.ts";
import {
  findQuote,
  findQuotes,
  quoteForSelection
} from "../../src/lib/artifacts/extensions/commentMarks.ts";
import { assetIdOf, isMediaFile } from "../../src/lib/artifacts/extensions/media.ts";
import { widgetSrcDoc } from "../../src/lib/artifacts/extensions/widgets.ts";
import { fromSaved, savedForm } from "../../src/lib/stores/useDocChatStore.ts";
import type { DbArtifactComment } from "../../src/lib/api/types.ts";

const comment = (o: Partial<DbArtifactComment> & { id: string }): DbArtifactComment => ({
  parent_id: null,
  author: "user",
  body: "x",
  quote: "",
  nth: 0,
  resolved: false,
  created_at: 1,
  updated_at: 1,
  ...o
});

describe("@comment operations", () => {
  it("are read like the other operations", () => {
    expect(parseOps("@comment 2\nWho owns this?\n@replace 1\nnew")).toEqual([
      { op: "comment", n: 2, text: "Who owns this?" },
      { op: "replace", n: 1, text: "new" }
    ]);
  });
  it("leave the text alone and are collected as questions", () => {
    const r = applyOps(["a", "b"], [{ op: "comment", n: 2, text: "Why?" }]);
    expect(r.blocks).toEqual(["a", "b"]);
    expect(r.questions).toEqual([{ n: 2, text: "Why?" }]);
  });
  it("a question about a block that does not exist is skipped like any bad number", () => {
    const r = applyOps(["a"], [{ op: "comment", n: 5, text: "Why?" }]);
    expect(r.questions).toEqual([]);
    expect(r.missing).toEqual([5]);
  });
  it("show up in the chat line", () => {
    expect(describeChange({ replaced: 1, inserted: 0, deleted: 0, questions: [1] })).toBe(
      "Changed 1 block, asked 1 question in the comments."
    );
    expect(describeChange({ replaced: 0, inserted: 0, deleted: 0, questions: [1, 2] })).toBe(
      "Asked 2 questions in the comments."
    );
  });
});

describe("plainQuote", () => {
  it("is the words a block starts with, without Markdown marks", () => {
    expect(plainQuote("## **Core** palette")).toBe("Core palette");
    expect(plainQuote("- [ ] ship [the beta](https://x.io) now")).toBe("ship the beta now");
    expect(plainQuote("> a quote\nsecond line")).toBe("a quote");
    expect(plainQuote("1. first item")).toBe("first item");
  });
  it("takes the first cell of a table and nothing for code", () => {
    expect(plainQuote("| Role | Light |\n| --- | --- |\n| a | b |")).toBe("Role");
    expect(plainQuote("```js\nlet a = 1;\n```")).toBe("");
  });
  it("is cut at a word near 80 characters", () => {
    const q = plainQuote("word ".repeat(40));
    expect(q.length).toBeLessThanOrEqual(80);
    expect(q.endsWith("word")).toBe(true);
  });
  it("counts earlier occurrences of the same words", () => {
    expect(quoteOccurrence(["Totals", "x", "Totals and more"], 2, "Totals")).toBe(1);
    expect(quoteOccurrence(["a"], 0, "a")).toBe(0);
  });
});

describe("comment threads", () => {
  const items = [
    comment({ id: "r2", created_at: 5, quote: "two" }),
    comment({ id: "r1", created_at: 1, author: "dara", quote: "one" }),
    comment({ id: "a", parent_id: "r1", created_at: 3 }),
    comment({ id: "b", parent_id: "r1", created_at: 2, author: "dara" })
  ];
  it("group replies under their root, oldest first", () => {
    const t = buildThreads(items);
    expect(t.map((x) => x.root.id)).toEqual(["r1", "r2"]);
    expect(t[0]!.replies.map((r) => r.id)).toEqual(["b", "a"]);
  });
  it("know when Dara is waited on", () => {
    const [t1, t2] = buildThreads(items);
    expect(awaitsDara(t1!)).toBe(true); // Dara took part, the user spoke last
    expect(awaitsDara(t2!)).toBe(false); // Dara never took part
  });
  it("read @dara as a request", () => {
    expect(mentionsDara("hey @dara fix this")).toBe(true);
    expect(mentionsDara("@Dara")).toBe(true);
    expect(mentionsDara("mail me at a@dara.io")).toBe(false);
  });
  it("go to Dara as plain text", () => {
    const text = threadForPrompt(buildThreads(items)[0]!);
    expect(text).toContain('Text the thread points at: "one"');
    expect(text).toContain("Dara: x");
    expect(text).toContain("User: x");
    expect(commentInstruction(buildThreads(items)[0]!)).toContain("needs your attention");
  });
});

describe("comment anchors in the editor", () => {
  const make = (html: string) => new Editor({ extensions: [StarterKit], content: html });
  it("finds every occurrence, never across blocks", () => {
    const ed = make("<p>two one two</p><p>two</p>");
    const hits = findQuotes(ed.state.doc, "two");
    expect(hits).toHaveLength(3);
    expect(ed.state.doc.textBetween(hits[2]!.from, hits[2]!.to)).toBe("two");
    expect(findQuotes(ed.state.doc, "two\ntwo")).toHaveLength(0);
    ed.destroy();
  });
  it("picks the nth occurrence, and the last when n is too big", () => {
    const ed = make("<p>a b a b a</p>");
    const second = findQuote(ed.state.doc, "a", 1)!;
    expect(second.from).toBeLessThan(findQuote(ed.state.doc, "a", 2)!.from);
    expect(findQuote(ed.state.doc, "a", 9)).toEqual(findQuote(ed.state.doc, "a", 2));
    expect(findQuote(ed.state.doc, "zzz", 0)).toBeNull();
    ed.destroy();
  });
  it("takes the selection as the anchor, with its occurrence number", () => {
    const ed = make("<p>go go go</p>");
    const second = findQuote(ed.state.doc, "go", 1)!;
    ed.commands.setTextSelection({ from: second.from, to: second.to });
    expect(quoteForSelection(ed)).toMatchObject({ quote: "go", nth: 1 });
    ed.destroy();
  });
  it("uses the whole paragraph when nothing is selected", () => {
    const ed = make("<p>first</p><p>second one</p>");
    ed.commands.setTextSelection(ed.state.doc.content.size - 3);
    expect(quoteForSelection(ed)?.quote).toBe("second one");
    ed.destroy();
  });
});

describe("saved chat messages", () => {
  it("drop the undo snapshot and keep what was said", () => {
    const saved = savedForm({
      id: "m1",
      role: "assistant",
      text: "Done.",
      steps: [{ label: "Plan", state: "done", time: "0:01" }],
      reasoning: "think",
      reasoningStreaming: true,
      thinkingMs: 900,
      status: "done",
      undo: { markdown: "big", title: "t" }
    });
    expect(saved).not.toHaveProperty("undo");
    expect(saved["reasoningStreaming"]).toBe(false);
    const back = fromSaved("m1", saved)!;
    expect(back).toMatchObject({ role: "assistant", text: "Done.", status: "done", undo: null });
  });
  it("mark a message that was still running when the app closed as interrupted", () => {
    const m = fromSaved("m2", { role: "assistant", status: "running", text: "" });
    expect(m).toMatchObject({ status: "stopped", text: "Interrupted when the app was closed." });
  });
  it("ignore rows that are not messages", () => {
    expect(fromSaved("m3", { role: "system" })).toBeNull();
    expect(fromSaved("m4", { role: "user", text: "hi" })).toEqual({
      id: "m4",
      role: "user",
      text: "hi"
    });
  });
});

describe("media and widgets", () => {
  it("only an asset stored with the document is ever loaded", () => {
    expect(assetIdOf("dara-asset:0F0F0F0F-1234-4abc-8def-0123456789ab")).toBe(
      "0f0f0f0f-1234-4abc-8def-0123456789ab"
    );
    expect(assetIdOf("https://example.com/a.png")).toBeNull();
    expect(assetIdOf("dara-asset:../../etc")).toBeNull();
  });
  it("accepts only images and video the server accepts", () => {
    expect(isMediaFile({ type: "image/png" })).toBe(true);
    expect(isMediaFile({ type: "video/webm" })).toBe(true);
    expect(isMediaFile({ type: "image/svg+xml" })).toBe(false);
    expect(isMediaFile({ type: "text/html" })).toBe(false);
  });
  it("a widget page blocks every outside request and cannot break out of its style variables", () => {
    const html = widgetSrcDoc("<b>x</b>", { text: "#111;}</style><script>", scheme: "dark" }, "w1");
    expect(html).toContain("default-src 'none'");
    expect(html).toContain("script-src 'unsafe-inline'");
    expect(html).not.toContain("</style><script>");
    expect(html).toContain("<b>x</b>");
    expect(html).toContain('"w1"');
  });
});

describe("widget guidance", () => {
  it("is given only when the user asks for something interactive", async () => {
    const { wantsWidget, buildOpsMessages, buildPlanMessages, WIDGET_GUIDE } =
      await import("../../src/lib/artifacts/plan.ts");
    expect(wantsWidget("add an interactive tip calculator")).toBe(true);
    expect(wantsWidget("make a WIDGET for this")).toBe(true);
    expect(wantsWidget("make it shorter")).toBe(false);
    expect(String(buildOpsMessages(["a"], "add a quiz")[0]!.content)).toContain(WIDGET_GUIDE);
    expect(String(buildOpsMessages(["a"], "shorten")[0]!.content)).not.toContain("```widget");
    expect(String(buildPlanMessages("a unit converter")[0]!.content)).toContain("one section");
  });
});

describe("widgets from the model", () => {
  it("turns an html block with a script into a widget, and leaves other code alone", async () => {
    const { normalizeWidgetFences } = await import("../../src/lib/artifacts/widgetFences.ts");
    const page = "```html\n<button id=a>Go</button>\n<script>a.onclick=()=>1</script>\n```";
    expect(normalizeWidgetFences(page)).toBe(page.replace("```html", "```widget"));
    const docs = "```html\n<p>just markup</p>\n```";
    expect(normalizeWidgetFences(docs)).toBe(docs);
    const js = "```js\nlet a = 1;\n```";
    expect(normalizeWidgetFences(js)).toBe(js);
    expect(normalizeWidgetFences("```widget\n<button>x</button>\n```")).toBe(
      "```widget\n<button>x</button>\n```"
    );
  });
  it("shows progress, not code, while a widget is still being written", async () => {
    const { previewWithProgress } = await import("../../src/lib/artifacts/widgetFences.ts");
    const half = "Intro.\n\n```widget\n<div>\n<button>Go</button>";
    expect(previewWithProgress(half)).toBe("Intro.\n\n*Building the widget... 2 lines written*");
    expect(previewWithProgress("```html\n")).toContain("Building the widget");
    const done = "```widget\n<b>x</b>\n```";
    expect(previewWithProgress(done)).toBe(done);
    expect(previewWithProgress("```python\nprint(1)")).toBe("```python\nprint(1)");
  });
});

describe("Word export", () => {
  it("builds a .docx (a zip) from the page's structure", async () => {
    const { buildDocx } = await import("../../src/lib/artifacts/export/docx.ts");
    const blob = await buildDocx(
      {
        type: "doc",
        content: [
          { type: "heading", attrs: { level: 1 }, content: [{ type: "text", text: "Title" }] },
          {
            type: "paragraph",
            content: [
              { type: "text", text: "bold", marks: [{ type: "bold" }] },
              { type: "text", text: " and a " },
              {
                type: "text",
                text: "link",
                marks: [{ type: "link", attrs: { href: "https://x.io" } }]
              }
            ]
          },
          {
            type: "table",
            content: [
              {
                type: "tableRow",
                content: [
                  {
                    type: "tableHeader",
                    content: [{ type: "paragraph", content: [{ type: "text", text: "A" }] }]
                  }
                ]
              }
            ]
          },
          {
            type: "codeBlock",
            attrs: { language: "widget" },
            content: [{ type: "text", text: "<b>x</b>" }]
          },
          { type: "image", attrs: { src: "https://evil.example/a.png", alt: "x" } }
        ]
      },
      { title: "Doc", loadImage: async () => null }
    );
    const head = new Uint8Array(
      await new Promise<ArrayBuffer>((resolve) => {
        const r = new FileReader();
        r.onload = () => resolve(r.result as ArrayBuffer);
        r.readAsArrayBuffer(blob.slice(0, 2));
      })
    );
    expect(String.fromCharCode(head[0]!, head[1]!)).toBe("PK");
    expect(blob.size).toBeGreaterThan(2000);
  });
});

describe("a 401 from the engine", () => {
  it("fetches the current key once and repeats the request with it", async () => {
    const { DaraApiClient } = await import("../../src/lib/api/client.ts");
    const seen: Array<string | null> = [];
    const realFetch = globalThis.fetch;
    globalThis.fetch = (async (_url: unknown, init?: RequestInit) => {
      const auth = new Headers(init?.headers).get("Authorization");
      seen.push(auth);
      return auth === "Bearer new"
        ? new Response(JSON.stringify({ ok: 1 }), { status: 200 })
        : new Response(JSON.stringify({ error: { message: "invalid or missing API key" } }), {
            status: 401
          });
    }) as typeof fetch;
    try {
      const client = new DaraApiClient({ baseUrl: "http://x", apiKey: "old" });
      let asked = 0;
      client.setAuthRefresher(async () => {
        asked++;
        client.setConnection("http://x", "new");
        return true;
      });
      await expect(client.request("/a")).resolves.toEqual({ ok: 1 });
      expect(seen).toEqual(["Bearer old", "Bearer new"]);
      expect(asked).toBe(1);
      // a refresher that has nothing new: the 401 is reported, not retried forever
      const stuck = new DaraApiClient({ baseUrl: "http://x", apiKey: "old" });
      stuck.setAuthRefresher(async () => false);
      await expect(stuck.request("/a")).rejects.toMatchObject({ status: 401 });
    } finally {
      globalThis.fetch = realFetch;
    }
  });
  it("is explained in words", async () => {
    const { collabMessage } = await import("../../src/lib/artifacts/errors.ts");
    const { ApiError } = await import("../../src/lib/api/errors.ts");
    expect(collabMessage(new ApiError(401, "x"), "comments")).toMatch(/key/);
  });
});
