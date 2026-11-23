import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ChatMessage } from "../../src/lib/api/types.ts";

// The model is a queue of answers: each request the assistant makes takes the next one, and the requests are kept to be looked at.
const answers: string[] = [];
const requests: ChatMessage[][] = [];
vi.mock("../../src/lib/artifacts/completion.ts", () => ({
  complete: async (run: { messages: ChatMessage[]; onContent?: (raw: string) => void }) => {
    requests.push(run.messages);
    const content = answers.shift() ?? "";
    run.onContent?.(content);
    return { content, reasoning: "", thinkingMs: 0, aborted: false };
  }
}));

import {
  deckFromAnswer,
  runAssistant,
  type AssistantRun
} from "../../src/lib/artifacts/assistant.ts";
import { parseDeck } from "../../src/lib/slides/deck.ts";

const hooks = () => {
  const seen: string[] = [];
  return {
    seen,
    hooks: {
      setDoc: (md: string) => void seen.push(md),
      setTitle: () => {},
      steps: () => {},
      reasoning: () => {}
    }
  };
};

const run = (
  markdown: string,
  instruction: string,
  h = hooks()
): AssistantRun & { seen: string[] } => ({
  kind: "slides",
  markdown,
  title: "",
  instruction,
  effort: "none",
  signal: new AbortController().signal,
  hooks: h.hooks,
  seen: h.seen
});

beforeEach(() => {
  answers.length = 0;
  requests.length = 0;
});

describe("the deck out of the model's answer", () => {
  it("drops a code fence, a lead-in line and a theme of its own", () => {
    expect(deckFromAnswer("```markdown\n# Dara\n\nHi\n\n---\n\n## One\n- a\n```")).toBe(
      "# Dara\n\nHi\n\n---\n\n## One\n- a"
    );
    expect(deckFromAnswer("[//]: # (theme: midnight)\n\n# Dara")).toBe("# Dara");
  });

  it("is nothing for an answer that is the model talking, not a deck", () => {
    expect(deckFromAnswer("")).toBe("");
    expect(deckFromAnswer("   \n")).toBe("");
    expect(deckFromAnswer("I am sorry, I cannot make a deck about that.")).toBe("");
    expect(deckFromAnswer("# Title only")).toBe("# Title only");
    expect(deckFromAnswer("first\n\n---\n\nsecond")).toBe("first\n\n---\n\nsecond");
  });

  it("works on a half-written answer", () => {
    expect(deckFromAnswer("# Dara\n\nHi\n\n---\n\n## On")).toBe("# Dara\n\nHi\n\n---\n\n## On");
  });
});

describe("writing a new deck", () => {
  it("writes the whole deck from one request and keeps the deck's look", async () => {
    answers.push(
      "# Dara\n\nAI on your own GPU\n\n---\n\n## Why local\n- Private\n- Fast\n\n---\n\n## Next steps\n- Try it",
      "I wrote a 3-slide deck."
    );
    const r = run("[//]: # (theme: ocean)", "A pitch for Dara");
    const out = await runAssistant(r);
    expect(out.changed).toBe(true);
    expect(out.stopped).toBe(false);
    expect(out.title).toBe("Dara");
    expect(out.summary).toBe('Wrote "Dara" in 3 slides.');
    expect(out.reply).toBe("I wrote a 3-slide deck.");
    expect(out.questions).toEqual([]);
    const d = parseDeck(out.markdown);
    expect([d.theme, d.slides.length]).toEqual(["ocean", 3]);
    // the page was shown the live text, with the look kept
    expect(r.seen.length).toBeGreaterThan(0);
    expect(r.seen.every((md) => md.startsWith("[//]: # (theme: ocean)"))).toBe(true);
    // the model was asked for a deck, and was not shown the theme line
    expect(requests[0]![0]!.content).toContain("slide editor");
    expect(JSON.stringify(requests[0])).not.toContain("theme:");
    expect(requests[0]![1]!.content).toBe("Write this deck: A pitch for Dara");
  });

  it("changes nothing when the model does not write a deck", async () => {
    answers.push("I am sorry, I cannot help with that.");
    const r = run("", "something");
    const out = await runAssistant(r);
    expect(out.changed).toBe(false);
    expect(out.summary).toMatch(/could not write that/);
    expect(out.markdown).toBe("");
  });
});

describe("changing an existing deck", () => {
  const deck =
    "[//]: # (theme: midnight)\n\n# Dara\n\nHi\n\n---\n\n## One\n\n- a\n- b\n\n---\n\n## Two\n\n- c";
  const heads = (md: string) => parseDeck(md).slides.map((x) => x.split("\n")[0]);

  it("numbers whole slides, applies the operations, keeps the look, and tells the model it is a deck", async () => {
    // slides: [1] # Dara + Hi   [2] ## One + list   [3] ## Two + list
    answers.push("@replace 2\n## One\n\n- a changed\n- b", "Changed one slide.");
    const r = run(deck, "change the first point on slide 2");
    const out = await runAssistant(r);
    expect(out.changed).toBe(true);
    const d = parseDeck(out.markdown);
    expect(d.theme).toBe("midnight");
    expect(d.slides).toEqual(["# Dara\n\nHi", "## One\n\n- a changed\n- b", "## Two\n\n- c"]);
    expect(requests[0]![0]!.content).toContain("slide editor");
    expect(requests[0]![0]!.content).toContain("@insert before N");
    expect(JSON.stringify(requests[0])).not.toContain("theme:");
    // the units are slides: three of them, with no --- lines to number
    expect(requests[0]![1]!.content).toContain("Slides:\n[1]\n# Dara\n\nHi\n\n[2]\n## One");
    expect(requests[0]![1]!.content).not.toContain("[4]");
    expect(out.summary).toBe("Changed 1 slide.");
    expect(out.reply).toBe("Changed one slide.");
  });

  it("adds a slide where it is told: after N, or before N", async () => {
    answers.push("@insert after 3\n## Three\n\n- d", "Added.");
    let out = await runAssistant(run(deck, "add a slide at the end"));
    expect(heads(out.markdown)).toEqual(["# Dara", "## One", "## Two", "## Three"]);
    expect(parseDeck(out.markdown).theme).toBe("midnight");

    answers.push("@insert before 3\n## Between\n\n- e", "Added.");
    out = await runAssistant(run(deck, "add a slide before the last one"));
    expect(heads(out.markdown)).toEqual(["# Dara", "## One", "## Between", "## Two"]);

    answers.push("@insert before 1\n## First\n\n- f", "Added.");
    out = await runAssistant(run(deck, "add a slide first"));
    const d = parseDeck(out.markdown);
    expect(d.slides[0]).toBe("## First\n\n- f");
    expect(d.slides.length).toBe(4);
  });

  it("removes a slide", async () => {
    answers.push("@delete 2", "Removed.");
    const out = await runAssistant(run(deck, "remove slide 2"));
    expect(parseDeck(out.markdown).slides).toEqual(["# Dara\n\nHi", "## Two\n\n- c"]);
  });

  it("ignores slides the model writes back after the change (a small model keeps going)", async () => {
    answers.push(
      "@replace 1\n# Dara: local AI\n\nHi\n@insert 0\n## One\n\n- a\n- b\n## Two\n\n- c\n@insert after 1\n# Dara: local AI\n\nHi",
      "Renamed."
    );
    const out = await runAssistant(run(deck, "rename the deck to Dara: local AI"));
    expect(parseDeck(out.markdown).slides).toEqual([
      "# Dara: local AI\n\nHi",
      "## One\n\n- a\n- b",
      "## Two\n\n- c"
    ]);
    expect(out.summary).toBe("Changed 1 slide.");
  });

  it("asks again once when the model names a slide that is not there, in the words of a deck", async () => {
    answers.push("@replace 9\n## Nine", "@replace 3\n## Two\n\n- c2", "Fixed.");
    const out = await runAssistant(run(deck, "change the last slide"));
    const retry = requests[1]![requests[1]!.length - 1]!.content;
    expect(retry).toContain("slide 9 does not exist");
    expect(retry).toContain("The deck has 3 slides, numbered 1 to 3");
    expect(parseDeck(out.markdown).slides[2]).toBe("## Two\n\n- c2");
  });

  it("never turns a question into a comment: a deck has none", async () => {
    answers.push("@comment 2\nWhich pricing do you want?", "");
    const out = await runAssistant(run(deck, "add pricing"));
    expect(out.questions).toEqual([]);
  });

  it("does not use the widget rules of a document, even for an interactive request", async () => {
    answers.push("@replace 2\n## One\n\n- a\n- b");
    await runAssistant(run(deck, "make it interactive with a quiz"));
    expect(requests[0]![0]!.content).not.toContain("widget");
  });
});
