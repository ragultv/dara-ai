<script lang="ts">
  import { marked } from "marked";
  import DOMPurify from "dompurify";

  interface Props {
    content: string;
    class?: string;
  }

  const { content, class: className = "" }: Props = $props();

  export function cleanLLMOutput(text: string): string {
    if (!text) return "";
    let r = text;
    // 1. Unicode table pipes
    r = r.replace(/│/g, "|").replace(/┃/g, "|");
    // 2. Fix inline complete code blocks: ```lang content``` on one line
    r = r.replace(/```(\w+) ((?:(?!```|\n).)+)```/g, "\n```$1\n$2\n```\n");
    // 3. Code fences not at start of line -> add newline before
    r = r.replace(/([^\n`])(`{3})/g, "$1\n$2");
    // 4. Closing fence not followed by newline -> add newline after
    r = r.replace(/(```)([ \t]*)([^\n`a-z])/g, "$1\n$3");
    // 5. Fix headers anywhere (not just line-start)
    r = r.replace(/(#{2,6})([^\s#\n])/g, "$1 $2");
    r = r.replace(/([^\n])(#{2,6} )/g, "$1\n$2");
    // 6. Numbered list items compressed after punctuation
    r = r.replace(/([.!?:)\]])[ \t]*(\d+\.[ \t])/g, "$1\n$2");
    // 7. Bullet list items compressed after colon
    r = r.replace(/:[ \t]*(-[ \t])/g, ":\n$1");
    // 8. Separator/rule fixes
    r = r.replace(/^(---+)(#{2,6} )/gm, "$1\n\n$2");
    r = r.replace(/(\n---+)(#{2,6} )/g, "$1\n\n$2");
    return r;
  }

  const customRenderer = new marked.Renderer();

  function highlightTokens(code: string): string {
    // Single-pass tokenizer to prevent cascading replacements of HTML tags/attributes
    const tokenRegex = /(\/\/[^\n]*|#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)|(\b\d+(?:\.\d+)?\b)|(\b(?:const|let|var|function|return|if|else|for|while|import|export|from|def|class|async|await|try|catch|pub|fn|struct|impl|match|select|where|insert|update|delete)\b)|(\b(?:true|false|null|undefined|None|True|False)\b)/g;

    return code.replace(tokenRegex, (match, comment, str, num, kw, bool) => {
      if (comment) return `<span class="tok-comment">${comment}</span>`;
      if (str) return `<span class="tok-string">${str}</span>`;
      if (num) return `<span class="tok-number">${num}</span>`;
      if (kw) return `<span class="tok-keyword">${kw}</span>`;
      if (bool) return `<span class="tok-bool">${bool}</span>`;
      return match;
    });
  }

  customRenderer.code = function ({ text, lang }: { text: string; lang?: string }) {
    const language = (lang || "code").toLowerCase();
    const cleanText = text.replace(/^\n+|\n+$/g, "");
    // Escape HTML entities safely
    const escaped = cleanText
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    const highlighted = highlightTokens(escaped);

    return `<div class="oprel-code-block"><div class="oprel-code-header"><span class="oprel-code-lang">${language}</span><button type="button" class="oprel-code-copy-btn" aria-label="Copy code block"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg><span class="copy-label">Copy code</span></button></div><div class="oprel-code-content"><pre><code class="language-${language}">${highlighted}</code></pre></div></div>`;
  };

  customRenderer.codespan = function ({ text }: { text: string }) {
    return `<code class="oprel-inline-code">${text}</code>`;
  };

  marked.setOptions({
    gfm: true,
    breaks: true,
    renderer: customRenderer
  });

  const sanitizedHtml = $derived.by(() => {
    if (!content) return "";
    const cleaned = cleanLLMOutput(content);
    const raw = marked.parse(cleaned, { async: false }) as string;
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
      ],
      ALLOWED_URI_REGEXP: /^(?:(?:https?|mailto):|data:image\/(?:png|jpeg|webp);)/i
    });
  });

  function handleContainerClick(event: MouseEvent) {
    const target = event.target as HTMLElement | null;
    if (!target) return;
    const btn = target.closest(".oprel-code-copy-btn");
    if (!btn) return;

    const block = btn.closest(".oprel-code-block");
    if (!block) return;

    const codeEl = block.querySelector("code");
    if (!codeEl) return;

    const code = codeEl.textContent ?? "";
    void navigator.clipboard.writeText(code);

    const label = btn.querySelector(".copy-label");
    if (label) {
      label.textContent = "Copied!";
      setTimeout(() => {
        if (label) label.textContent = "Copy code";
      }, 2000);
    }
  }
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<!-- svelte-ignore a11y_click_events_have_key_events -->
<div class="oprel-markdown {className}" onclick={handleContainerClick}>
  <!-- Audited single HTML injection point for sanitized Markdown per docs/ui/06-coding-standards.md -->
  <!-- eslint-disable-next-line svelte/no-at-html-tags -->
  {@html sanitizedHtml}
</div>
