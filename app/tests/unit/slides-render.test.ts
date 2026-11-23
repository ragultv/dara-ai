import { describe, expect, it } from "vitest";
import { renderSlideHtml } from "../../src/lib/slides/SlideSurface.tsx";
import { slidePageHtml } from "../../src/lib/slides/printDeck.ts";

const ID = "0f8fad5b-d9cb-469f-a165-70867728950e";

describe("a slide written by the model is safe to draw", () => {
  it("draws the usual Markdown", () => {
    const html = renderSlideHtml("## Plan\n\n- **one**\n- two\n\n> quote\n\n`code`");
    expect(html).toContain("<h2>Plan</h2>");
    expect(html).toContain("<strong>one</strong>");
    expect(html).toContain("<li>two</li>");
    expect(html).toContain("<blockquote>");
    expect(html).toContain("<code>code</code>");
  });

  it("never lets raw HTML through: it is shown as text", () => {
    const html = renderSlideHtml(
      '<script>alert(1)</script>\n\n<img src=x onerror="alert(2)">\n\n<a href="javascript:alert(3)">x</a>'
    );
    // no real tag of any of those, only their text: the words "onerror" and "javascript:" are visible, escaped, never run
    expect(html).not.toMatch(/<(script|img|a)[\s>]/i);
    expect(html).toContain("&lt;script&gt;");
    expect(html).toContain("&lt;img");
    const dom = document.createElement("div");
    dom.innerHTML = html;
    expect(dom.querySelector("script, img, a, [onerror]")).toBeNull();
  });

  it("shows a link as text, so a click on a slide never leaves the app", () => {
    const html = renderSlideHtml("[the docs](https://example.com/docs)");
    expect(html).toContain('<span class="slide-link">the docs</span>');
    expect(html).not.toContain("href");
    expect(html).not.toContain("example.com");
  });

  it("draws only images stored with the deck, and drops any other (nothing leaves this computer)", () => {
    const mine = renderSlideHtml(`![chart](dara-asset:${ID})`);
    expect(mine).toContain(`data-asset="${ID}"`);
    expect(mine).not.toContain("src=");
    expect(renderSlideHtml("![tracker](https://evil.example/pixel.png)")).not.toContain("<img");
    expect(renderSlideHtml("![x](data:image/png;base64,AAAA)")).not.toContain("<img");
    expect(renderSlideHtml("![x](dara-asset:not-an-id)")).not.toContain("<img");
  });

  it("keeps tables and task lists", () => {
    const html = renderSlideHtml("| a | b |\n| - | - |\n| 1 | 2 |\n\n- [x] done\n- [ ] todo");
    expect(html).toContain("<table>");
    expect(html).toContain("<td>1</td>");
    expect(html).toContain('type="checkbox"');
  });
});

describe("a slide on the printed page", () => {
  it("is one page with the deck's look and its layout", () => {
    const html = slidePageHtml("# Dara\n\nRuns on your GPU", "midnight");
    expect(html).toContain("print-deck-page");
    expect(html).toContain("slide-theme-midnight");
    expect(html).toContain("slide-title");
    expect(html).toContain("<h1>Dara</h1>");
    expect(slidePageHtml("## A\n\n- b\n- c", "paper")).toContain("slide-content");
  });
});
