<script lang="ts">
  interface Props {
    code: string;
    language?: string;
    filename?: string;
    class?: string;
  }

  const { code, language = "", filename = "", class: className = "" }: Props = $props();

  let copied = $state(false);

  function copyCode() {
    void navigator.clipboard.writeText(code);
    copied = true;
    setTimeout(() => {
      copied = false;
    }, 2000);
  }
</script>

<div
  class="relative rounded-lg overflow-hidden border border-border bg-code-bg font-mono text-xs text-left {className}"
>
  {#if filename || language}
    <div
      class="flex items-center justify-between px-3.5 py-1.5 border-b border-border bg-surface-sunken text-text-subtle select-none"
    >
      <span class="truncate">{filename || language}</span>
      <button
        type="button"
        aria-label="Copy code block"
        onclick={copyCode}
        class="text-xs font-sans text-text-muted hover:text-text cursor-pointer focus-visible:outline-2 focus-visible:outline-focus"
      >
        {copied ? "Copied!" : "Copy"}
      </button>
    </div>
  {:else}
    <div class="absolute top-2 right-2 z-10">
      <button
        type="button"
        aria-label="Copy code block"
        onclick={copyCode}
        class="px-2 py-1 rounded bg-surface border border-border text-[11px] font-sans text-text-muted hover:text-text shadow-sm cursor-pointer focus-visible:outline-2 focus-visible:outline-focus"
      >
        {copied ? "Copied!" : "Copy"}
      </button>
    </div>
  {/if}

  <div class="p-4 overflow-x-auto">
    <pre class="leading-relaxed text-text font-mono"><code>{code}</code></pre>
  </div>
</div>
