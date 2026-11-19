<script lang="ts">
  import type { Snippet } from "svelte";

  interface Props {
    isOpen: boolean;
    title: string;
    description?: string;
    onclose: () => void;
    children?: Snippet;
    actions?: Snippet;
  }

  const { isOpen, title, description, onclose, children, actions }: Props = $props();

  const titleId = `dialog-title-${Math.random().toString(36).substring(2, 9)}`;
  const descId = `dialog-desc-${Math.random().toString(36).substring(2, 9)}`;

  let dialogElement: HTMLDivElement | null = $state(null);

  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === "Escape" && isOpen) {
      e.stopPropagation();
      onclose();
    }
  }

  $effect(() => {
    if (!isOpen) return;

    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", handleKeyDown);
    // Focus trap first focusable
    setTimeout(() => {
      const focusable = dialogElement?.querySelectorAll<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      if (focusable && focusable.length > 0) {
        focusable[0]?.focus();
      }
    }, 50);

    return () => {
      document.body.style.overflow = "";
      window.removeEventListener("keydown", handleKeyDown);
    };
  });
</script>

{#if isOpen}
  <!-- Backdrop -->
  <div
    class="fixed inset-0 z-modal bg-overlay flex items-center justify-center p-4 backdrop-blur-xs transition-opacity"
    role="presentation"
    onclick={(e) => {
      if (e.target === e.currentTarget) onclose();
    }}
  >
    <!-- Modal Card -->
    <div
      bind:this={dialogElement}
      role="dialog"
      aria-modal="true"
      aria-labelledby={titleId}
      aria-describedby={description ? descId : undefined}
      class="w-full max-w-md rounded-xl bg-surface-raised border border-border shadow-lg p-6 space-y-4 text-left transition-all"
    >
      <div class="flex items-start justify-between">
        <div class="space-y-1">
          <h2 id={titleId} class="text-lg font-semibold text-text">
            {title}
          </h2>
          {#if description}
            <p id={descId} class="text-xs text-text-muted">
              {description}
            </p>
          {/if}
        </div>

        <button
          type="button"
          aria-label="Close dialog"
          onclick={onclose}
          class="p-1 rounded-sm text-text-muted hover:text-text hover:bg-surface-sunken focus-visible:outline-2 focus-visible:outline-focus cursor-pointer"
        >
          <svg
            class="w-5 h-5"
            viewBox="0 0 20 20"
            fill="none"
            stroke="currentColor"
            stroke-width="1.5"
            aria-hidden="true"
          >
            <path d="M5 5l10 10M15 5L5 15" stroke-linecap="round" stroke-linejoin="round" />
          </svg>
        </button>
      </div>

      {#if children}
        <div class="text-sm text-text">
          {@render children()}
        </div>
      {/if}

      {#if actions}
        <div class="flex items-center justify-end space-x-3 pt-3 border-t border-border">
          {@render actions()}
        </div>
      {/if}
    </div>
  </div>
{/if}
