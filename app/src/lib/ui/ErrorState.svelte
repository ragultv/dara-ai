<script lang="ts">
  import Button from "./Button.svelte";

  interface Props {
    title?: string;
    message: string;
    details?: string;
    onretry?: () => void;
    class?: string;
  }

  const {
    title = "Something went wrong",
    message,
    details,
    onretry,
    class: className = ""
  }: Props = $props();

  let copied = $state(false);

  function copyDetails() {
    if (details) {
      void navigator.clipboard.writeText(details);
      copied = true;
      setTimeout(() => {
        copied = false;
      }, 2000);
    }
  }
</script>

<div
  role="alert"
  class="w-full p-6 rounded-xl bg-danger-subtle border border-danger text-left space-y-4 {className}"
>
  <div class="flex items-start space-x-3">
    <div class="p-2 rounded-lg bg-surface text-danger shrink-0 mt-0.5">
      <svg
        class="w-5 h-5"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        aria-hidden="true"
      >
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="8" x2="12" y2="12" />
        <line x1="12" y1="16" x2="12.01" y2="16" />
      </svg>
    </div>

    <div class="space-y-1">
      <h3 class="text-sm font-semibold text-text">{title}</h3>
      <p class="text-xs text-text-muted">{message}</p>
    </div>
  </div>

  {#if details}
    <div class="p-3 rounded-md bg-surface border border-border text-xs font-mono overflow-x-auto">
      <pre class="text-text-muted">{details}</pre>
    </div>
  {/if}

  <div class="flex items-center space-x-3 pt-2">
    {#if onretry}
      <Button variant="danger" size="sm" onclick={onretry}>Try again</Button>
    {/if}

    {#if details}
      <Button variant="secondary" size="sm" onclick={copyDetails}>
        {copied ? "Copied!" : "Copy details"}
      </Button>
    {/if}
  </div>
</div>
