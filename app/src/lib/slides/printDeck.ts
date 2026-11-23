import { assetIdOf, assetUrl } from "../artifacts/extensions/media.ts";
import { renderSlideHtml } from "./SlideSurface.tsx";
import { slideLayout, type Deck } from "./deck.ts";

// PDF export of a deck: one 16:9 page per slide. The slides are placed in #print-root as plain HTML (the print stylesheet in app.css
// hides the rest of the app and sets the page size), the images are loaded first, and the system print dialog saves it as a PDF.

const ROOT_ID = "print-root";
const STYLE_ID = "print-deck-style";

async function loadImages(root: HTMLElement, docId: string): Promise<void> {
  const jobs = Array.from(root.querySelectorAll<HTMLImageElement>("img[data-asset]")).map(
    async (img) => {
      const id = assetIdOf(`dara-asset:${img.dataset["asset"] ?? ""}`);
      if (!id) return img.remove();
      try {
        img.src = await assetUrl(docId, id);
        await img.decode();
      } catch {
        img.replaceWith(document.createTextNode("(image not found)"));
      }
    }
  );
  await Promise.all(jobs);
}

/** One slide as the HTML the print page holds (the same classes as the editor, so it looks the same). */
export function slidePageHtml(markdown: string, theme: string): string {
  return (
    `<div class="print-deck-page"><div class="slide slide-theme-${theme} slide-${slideLayout(markdown)}">` +
    `<div class="slide-body"><div class="slide-html">${renderSlideHtml(markdown)}</div></div></div></div>`
  );
}

/** Opens the print dialog for the deck. Resolves once the dialog has closed. */
export async function printDeck(deck: Deck, title: string, docId: string): Promise<void> {
  document.getElementById(ROOT_ID)?.remove();
  document.getElementById(STYLE_ID)?.remove();
  const root = document.createElement("div");
  root.id = ROOT_ID;
  root.innerHTML = deck.slides.map((s) => slidePageHtml(s, deck.theme)).join("");
  document.body.append(root);
  // the page is the size of one slide, with no margin (this rule comes after the document's own A4 rule, so it wins)
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = "@media print { @page { size: 1280px 720px; margin: 0; } }";
  document.head.append(style);
  await loadImages(root, docId);

  const previousTitle = document.title;
  document.title = title.trim() || "Untitled"; // the PDF's suggested file name
  return new Promise((resolve) => {
    const done = () => {
      window.removeEventListener("afterprint", done);
      document.title = previousTitle;
      root.remove();
      style.remove();
      resolve();
    };
    window.addEventListener("afterprint", done);
    requestAnimationFrame(() => {
      window.print();
      setTimeout(done, 120_000);
    });
  });
}
