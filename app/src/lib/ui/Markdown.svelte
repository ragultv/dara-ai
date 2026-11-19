<script lang="ts">
  import { marked } from "marked";
  import DOMPurify from "dompurify";

  interface Props {
    content: string;
    class?: string;
  }

  const { content, class: className = "" }: Props = $props();

  // Configure marked options
  marked.setOptions({
    gfm: true,
    breaks: true
  });

  const sanitizedHtml = $derived.by(() => {
    if (!content) return "";
    const raw = marked.parse(content, { async: false }) as string;
    // Sanitize with DOMPurify per docs/ui/05-security.md
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
        "hr"
      ],
      ALLOWED_ATTR: ["href", "src", "alt", "title", "target", "rel", "class"],
      ALLOWED_URI_REGEXP: /^(?:(?:https?|mailto):|data:image\/(?:png|jpeg|webp);)/i
    });
  });
</script>

<div class="prose prose-sm max-w-none text-text leading-relaxed text-left {className}">
  <!-- Audited single HTML injection point for sanitized Markdown per docs/ui/06-coding-standards.md -->
  <!-- eslint-disable-next-line svelte/no-at-html-tags -->
  {@html sanitizedHtml}
</div>
