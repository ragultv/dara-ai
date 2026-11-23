import React, { useEffect, useMemo, useRef } from "react";
import { Marked } from "marked";
import DOMPurify from "dompurify";
import { assetIdOf, assetUrl } from "../artifacts/extensions/media.ts";
import { slideLayout, type ThemeId } from "./deck.ts";

// One slide as a page. The same component is the thumbnail, the editing page (its children are the editor) and the full screen, because
// every size inside it is in container units. A slide written by the model is never trusted: raw HTML is sanitised, links are shown as
// text, and an image is drawn only when it is stored with this deck, so a slide can never contact a server.

const md = new Marked({ gfm: true, breaks: false });

md.use({
  renderer: {
    // an image of this deck, loaded after the HTML is in the page (the request carries the API key); any other image is dropped
    image({ href, text }) {
      const id = assetIdOf(href ?? "");
      return id ? `<img data-asset="${id}" alt="${escapeAttr(text ?? "")}">` : "";
    },
    link({ tokens }) {
      return `<span class="slide-link">${this.parser.parseInline(tokens)}</span>`;
    },
    html({ text }) {
      return escapeText(text);
    }
  }
});

function escapeAttr(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}
function escapeText(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

const ALLOWED_TAGS = [
  "p",
  "br",
  "strong",
  "em",
  "del",
  "code",
  "pre",
  "blockquote",
  "ul",
  "ol",
  "li",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "table",
  "thead",
  "tbody",
  "tr",
  "th",
  "td",
  "img",
  "hr",
  "span",
  "input"
];

/** The slide's Markdown as safe HTML. */
export function renderSlideHtml(markdown: string): string {
  const raw = md.parse(markdown, { async: false }) as string;
  return DOMPurify.sanitize(raw, {
    ALLOWED_TAGS,
    ALLOWED_ATTR: ["class", "alt", "data-asset", "type", "checked", "disabled"],
    ALLOW_DATA_ATTR: true
  });
}

/** Fills in the images of a rendered slide (blob URLs of the deck's own assets). */
function useAssetImages(root: React.RefObject<HTMLElement | null>, docId: string, html: string) {
  useEffect(() => {
    const el = root.current;
    if (!el) return;
    let live = true;
    el.querySelectorAll<HTMLImageElement>("img[data-asset]").forEach((img) => {
      const id = img.dataset["asset"];
      if (!id || img.getAttribute("src")) return;
      assetUrl(docId, id).then(
        (url) => live && img.setAttribute("src", url),
        () => live && img.replaceWith(document.createTextNode("(image not found)"))
      );
    });
    return () => {
      live = false;
    };
  }, [root, docId, html]);
}

interface Props {
  /** the slide's Markdown; sets the layout (title, section or content) */
  markdown: string;
  theme: ThemeId;
  /** the deck the slide's images belong to */
  docId: string;
  className?: string;
  /** a slide being edited puts the editor here; the slide then lays itself out from `markdown`, which the editor keeps current */
  children?: React.ReactNode;
}

export const SlideSurface: React.FC<Props> = ({
  markdown,
  theme,
  docId,
  className = "",
  children
}) => {
  const layout = slideLayout(markdown);
  const html = useMemo(() => (children ? "" : renderSlideHtml(markdown)), [markdown, children]);
  const ref = useRef<HTMLDivElement>(null);
  useAssetImages(ref, docId, html);
  return (
    <div className={`slide slide-theme-${theme} slide-${layout} ${className}`}>
      <div className="slide-body">
        {children ?? (
          <div ref={ref} className="slide-html" dangerouslySetInnerHTML={{ __html: html }} />
        )}
      </div>
    </div>
  );
};
