import React, { useMemo } from "react";
import { marked } from "marked";
import DOMPurify from "dompurify";

interface MarkdownProps {
  content: string;
  className?: string;
}

function escapeHtml(str: string): string {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function highlightTokens(rawCode: string): string {
  const tokenRegex =
    /(\/\/[^\n]*|#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)|(\b\d+(?:\.\d+)?\b)|(\b(?:const|let|var|function|return|if|else|for|while|import|export|from|def|class|async|await|try|catch|pub|fn|struct|impl|match|select|where|insert|update|delete)\b)|(\b(?:true|false|null|undefined|None|True|False)\b)/g;

  let lastIndex = 0;
  let html = "";
  let match: RegExpExecArray | null;

  while ((match = tokenRegex.exec(rawCode)) !== null) {
    // Append preceding plain text (escaped)
    if (match.index > lastIndex) {
      html += escapeHtml(rawCode.slice(lastIndex, match.index));
    }
    const [full, comment, str, num, kw, bool] = match;
    const escapedVal = escapeHtml(full);

    if (comment) {
      html += `<span class="tok-comment">${escapedVal}</span>`;
    } else if (str) {
      html += `<span class="tok-string">${escapedVal}</span>`;
    } else if (num) {
      html += `<span class="tok-number">${escapedVal}</span>`;
    } else if (kw) {
      html += `<span class="tok-keyword">${escapedVal}</span>`;
    } else if (bool) {
      html += `<span class="tok-bool">${escapedVal}</span>`;
    } else {
      html += escapedVal;
    }
    lastIndex = match.index + full.length;
  }

  if (lastIndex < rawCode.length) {
    html += escapeHtml(rawCode.slice(lastIndex));
  }

  return html;
}

const customRenderer = new marked.Renderer();

customRenderer.code = function ({ text, lang }: { text: string; lang?: string }) {
  const language = (lang || "code").toLowerCase();
  const cleanText = text.replace(/^\n+|\n+$/g, "");
  const highlighted = highlightTokens(cleanText);

  return `<div class="oprel-code-block"><div class="oprel-code-header"><span class="oprel-code-lang">${language}</span><button type="button" class="oprel-code-copy-btn" aria-label="Copy code block"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg><span class="copy-label">Copy code</span></button></div><div class="oprel-code-content"><pre><code class="language-${language}">${highlighted}</code></pre></div></div>`;
};

// A color written as inline code (#7C3AED) gets a small swatch, as in Claude's replies. Only a plain hex value ever reaches a style attribute.
const HEX_COLOR = /^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$/;

customRenderer.codespan = function ({ text }: { text: string }) {
  const plain = text.trim();
  const swatch = HEX_COLOR.test(plain)
    ? `<span class="hex-swatch" style="background:${plain}"></span>`
    : "";
  return `<code class="oprel-inline-code">${swatch}${escapeHtml(text)}</code>`;
};

// Model output must never carry styling of its own: a style attribute survives only on our own swatch, and only as a background color.
DOMPurify.addHook("uponSanitizeAttribute", (node, data) => {
  if (data.attrName !== "style") return;
  const ok =
    node instanceof Element &&
    node.classList.contains("hex-swatch") &&
    /^background:\s*#[0-9a-fA-F]{3,8}$/.test(data.attrValue);
  if (!ok) data.keepAttr = false;
});

marked.setOptions({
  gfm: true,
  breaks: true,
  renderer: customRenderer
});

export const Markdown: React.FC<MarkdownProps> = ({ content, className = "" }) => {
  const sanitizedHtml = useMemo(() => {
    if (!content) return "";
    const raw = marked.parse(content, { async: false }) as string;
    return DOMPurify.sanitize(raw, {
      ALLOWED_TAGS: [
        "p",
        "br",
        "strong",
        "em",
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
        "a",
        "img",
        "hr",
        "div",
        "span",
        "button",
        "svg",
        "rect",
        "path",
        "line",
        "circle"
      ],
      ALLOWED_ATTR: [
        "href",
        "src",
        "alt",
        "title",
        "target",
        "rel",
        "class",
        "style",
        "type",
        "aria-label",
        "width",
        "height",
        "viewBox",
        "fill",
        "stroke",
        "stroke-width",
        "stroke-linecap",
        "stroke-linejoin",
        "d",
        "rx",
        "ry",
        "x",
        "y",
        "x1",
        "x2",
        "y1",
        "y2",
        "cx",
        "cy",
        "r"
      ]
    });
  }, [content]);

  const handleContainerClick = (e: React.MouseEvent) => {
    const target = e.target as HTMLElement | null;
    if (!target) return;
    const btn = target.closest(".oprel-code-copy-btn");
    if (!btn) return;
    const block = btn.closest(".oprel-code-block");
    if (!block) return;
    const codeEl = block.querySelector("code");
    if (!codeEl) return;
    void navigator.clipboard.writeText(codeEl.textContent ?? "");
    const label = btn.querySelector(".copy-label");
    if (label) {
      label.textContent = "Copied!";
      setTimeout(() => {
        if (label) label.textContent = "Copy code";
      }, 2000);
    }
  };

  return (
    <div
      className={`oprel-markdown ${className}`}
      onClick={handleContainerClick}
      dangerouslySetInnerHTML={{ __html: sanitizedHtml }}
    />
  );
};
