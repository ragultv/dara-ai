<script lang="ts">
  import Icon from "./Icon.svelte";

  interface Props {
    code: string;
    language?: string;
    filename?: string;
    class?: string;
  }

  const { code, language = "", filename = "", class: className = "" }: Props = $props();

  let copied = $state(false);
  const trimmedCode = $derived(code ? code.trim() : "");

  function copyCode() {
    void navigator.clipboard.writeText(code);
    copied = true;
    setTimeout(() => {
      copied = false;
    }, 2000);
  }
</script>

<div class="oprel-code-block {className}">
  <div class="oprel-code-header">
    <span class="oprel-code-lang">{filename || language || "code"}</span>
    <button
      type="button"
      class="oprel-code-copy-btn"
      aria-label="Copy code block"
      onclick={copyCode}
    >
      {#if copied}
        <Icon name="check" size={12} class="text-success" />
        <span class="text-success">Copied!</span>
      {:else}
        <Icon name="copy" size={12} />
        <span>Copy code</span>
      {/if}
    </button>
  </div>

  <div class="oprel-code-content">
    <pre class="leading-relaxed font-mono"><code>{trimmedCode}</code></pre>
  </div>
</div>
