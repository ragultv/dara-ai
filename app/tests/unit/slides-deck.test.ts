import { describe, expect, it } from "vitest";
import {
  addSlide,
  clampIndex,
  deckIsEmpty,
  deleteSlide,
  dropRepeats,
  duplicateSlide,
  firstSlide,
  moveSlide,
  parseDeck,
  replaceSlide,
  serializeDeck,
  slideLayout,
  slideTitle
} from "../../src/lib/slides/deck.ts";

describe("reading a deck", () => {
  it("splits slides at a --- line between blank lines", () => {
    const d = parseDeck("# One\n\nhello\n\n---\n\n## Two\n- a\n- b\n\n---\n\n## Three");
    expect(d.slides).toEqual(["# One\n\nhello", "## Two\n- a\n- b", "## Three"]);
    expect(d.theme).toBe("paper");
  });

  it("does not split inside a code block, or at the underline of a heading", () => {
    const code = "## Code\n\n```md\ntext\n\n---\n\nmore\n```";
    expect(parseDeck(`${code}\n\n---\n\n## Next`).slides).toEqual([code, "## Next"]);
    const setext = "Heading\n---\n\ntext";
    expect(parseDeck(setext).slides).toEqual([setext]);
  });

  it("copes with Windows line ends, extra blank lines and a --- at the start or end", () => {
    expect(parseDeck("# A\r\n\r\n---\r\n\r\n# B\r\n").slides).toEqual(["# A", "# B"]);
    expect(parseDeck("\n\n---\n\n# A\n\n\n\n---\n\n").slides).toEqual(["", "# A", ""]);
  });

  it("is one blank slide for no text", () => {
    expect(parseDeck("").slides).toEqual([""]);
    expect(parseDeck("  \n \n").slides).toEqual([""]);
    expect(deckIsEmpty("")).toBe(true);
    expect(deckIsEmpty("[//]: # (theme: ocean)")).toBe(true);
    expect(deckIsEmpty("# Hi")).toBe(false);
  });

  it("reads the theme line, and ignores one it does not know", () => {
    expect(parseDeck("[//]: # (theme: midnight)\n\n# A\n\n---\n\n# B")).toEqual({
      theme: "midnight",
      slides: ["# A", "# B"]
    });
    const odd = parseDeck("[//]: # (theme: neon)\n\n# A");
    expect(odd.theme).toBe("paper");
    expect(odd.slides).toEqual(["# A"]);
    // a theme line anywhere but the start is just a (hidden) line of that slide
    expect(parseDeck("# A\n\n---\n\n[//]: # (theme: ocean)").theme).toBe("paper");
  });
});

describe("writing a deck", () => {
  it("joins slides with a --- line, and round-trips", () => {
    const md = "[//]: # (theme: ocean)\n\n# One\n\n---\n\n## Two\n- a";
    expect(serializeDeck(parseDeck(md))).toBe(md);
    expect(serializeDeck(parseDeck("# A\n\n---\n\n# B"))).toBe("# A\n\n---\n\n# B");
  });

  it("leaves the default theme out, and an empty deck empty", () => {
    expect(serializeDeck({ theme: "paper", slides: ["# A"] })).toBe("# A");
    expect(serializeDeck({ theme: "paper", slides: [""] })).toBe("");
    expect(serializeDeck({ theme: "sunrise", slides: [""] })).toBe("[//]: # (theme: sunrise)");
    // two blank slides are kept: they are a deck of two slides
    expect(parseDeck(serializeDeck({ theme: "paper", slides: ["", ""] })).slides).toEqual(["", ""]);
  });
});

describe("changing a deck", () => {
  const deck = { theme: "paper" as const, slides: ["# A", "# B", "# C"] };

  it("replaces, adds, duplicates and deletes slides", () => {
    expect(replaceSlide(deck, 1, "  # Bee \n").slides).toEqual(["# A", "# Bee", "# C"]);
    expect(replaceSlide(deck, 9, "x")).toBe(deck);
    expect(addSlide(deck, 0).slides).toEqual(["# A", "", "# B", "# C"]);
    expect(addSlide(deck, 2).slides).toEqual(["# A", "# B", "# C", ""]);
    expect(addSlide(deck, 99).slides).toEqual(["# A", "# B", "# C", ""]);
    expect(duplicateSlide(deck, 1).slides).toEqual(["# A", "# B", "# B", "# C"]);
    expect(deleteSlide(deck, 1).slides).toEqual(["# A", "# C"]);
    expect(deleteSlide({ ...deck, slides: ["# A"] }, 0).slides).toEqual([""]);
  });

  it("moves a slide to a position", () => {
    expect(moveSlide(deck, 0, 2).slides).toEqual(["# B", "# C", "# A"]);
    expect(moveSlide(deck, 2, 0).slides).toEqual(["# C", "# A", "# B"]);
    expect(moveSlide(deck, 1, 1)).toBe(deck);
    expect(moveSlide(deck, 0, 3)).toBe(deck);
  });

  it("keeps an index inside the deck", () => {
    expect(clampIndex(deck, -3)).toBe(0);
    expect(clampIndex(deck, 7)).toBe(2);
    expect(clampIndex(deck, 1)).toBe(1);
  });
});

describe("naming and laying out a slide", () => {
  it("names a slide by its first heading, else its first words", () => {
    expect(slideTitle("# **Launch** plan\n\n- a")).toBe("Launch plan");
    expect(slideTitle("Some text first\n\n## Later heading")).toBe("Later heading");
    expect(slideTitle("- first point\n- second")).toBe("first point");
    expect(slideTitle("![pic](dara-asset:x)\n\n[link](http://a.b)")).toBe("link");
    expect(slideTitle("")).toBe("");
    expect(slideTitle(`# ${"word ".repeat(30)}`).length).toBeLessThanOrEqual(60);
  });

  it("centres a title slide and a section slide, and writes the rest from the top", () => {
    expect(slideLayout("# Dara\n\nRuns on your own GPU")).toBe("title");
    expect(slideLayout("# Dara")).toBe("title");
    expect(slideLayout("## Part two")).toBe("section");
    expect(slideLayout("# Dara\n\n- a\n- b")).toBe("content");
    expect(slideLayout("# Dara\n\none\n\ntwo")).toBe("content");
    expect(slideLayout("## Heading\n\ntext")).toBe("content");
    expect(slideLayout("just text")).toBe("content");
    expect(slideLayout("")).toBe("content");
  });

  it("gives the gallery the first slide and the look", () => {
    expect(firstSlide("[//]: # (theme: ocean)\n\n# A\n\n---\n\n# B")).toEqual({
      theme: "ocean",
      slide: "# A"
    });
  });
});

describe("an edit that only repeats something", () => {
  const slides = [
    "# Dara\n\nAI that runs on your own GPU",
    "## Why local\n\n- Private\n- Fast",
    "## Next steps\n\n- Install Dara"
  ];

  it("drops an insert that pastes back a slide already in the deck", () => {
    const ops = [
      { op: "replace", n: 1, text: "# Dara: local AI\n\nAI that runs on your own GPU" },
      {
        op: "insert",
        n: 0,
        text: "## Why local\n\n- Private\n- Fast\n## Next steps\n\n- Install Dara"
      }
    ];
    expect(dropRepeats(ops, slides)).toEqual([ops[0]]);
  });

  it("ignores spacing and capitals when it compares", () => {
    const ops = [{ op: "insert", n: 1, text: "##  WHY   LOCAL\n\n-  private\n-  FAST" }];
    expect(dropRepeats(ops, slides)).toEqual([]);
  });

  it("drops an insert that repeats the text of the operation before it", () => {
    const ops = [
      { op: "replace", n: 2, text: "## Why local\n\n- Private" },
      { op: "insert", n: 2, text: "## Why local\n\n- Private" }
    ];
    expect(dropRepeats(ops, ["# A\n\nsub", "## Why local\n\n- Private\n- Fast"])).toEqual([ops[0]]);
  });

  it("keeps a genuinely new slide, a delete, and an insert that only shares a short phrase", () => {
    const ops = [
      { op: "insert", n: 1, text: "## Pricing\n\n- Free to start" },
      { op: "delete", n: 3 },
      { op: "insert", n: 2, text: "## Why local, again\n\n- Because it is yours" }
    ];
    expect(dropRepeats(ops, slides)).toEqual(ops);
  });

  it("does not guard a very short slide", () => {
    expect(
      dropRepeats([{ op: "insert", n: 1, text: "## Hi there, everyone" }], ["Hi"])
    ).toHaveLength(1);
  });
});
