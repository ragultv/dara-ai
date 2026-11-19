<script lang="ts">
  import type { Snippet } from "svelte";

  interface Props {
    title: string;
    description?: string;
    icon?: Snippet;
    actions?: Snippet;
    class?: string;
  }

  const { title, description, icon, actions, class: className = "" }: Props = $props();
</script>

<div
  class="flex flex-col items-center justify-center p-8 text-center max-w-sm mx-auto space-y-4 {className}"
>
  {#if icon}
    <div class="text-primary opacity-80">
      {@render icon()}
    </div>
  {:else}
    <!-- Default geometric mark -->
    <div
      class="w-12 h-12 rounded-xl bg-primary-subtle flex items-center justify-center text-primary"
    >
      <svg
        class="w-6 h-6"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="1.5"
        aria-hidden="true"
      >
        <rect x="3" y="3" width="18" height="18" rx="4" />
        <path d="M9 9h6M9 13h4" stroke-linecap="round" />
      </svg>
    </div>
  {/if}

  <div class="space-y-1">
    <h3 class="text-base font-semibold text-text">{title}</h3>
    {#if description}
      <p class="text-xs text-text-muted leading-relaxed">{description}</p>
    {/if}
  </div>

  {#if actions}
    <div class="pt-2">
      {@render actions()}
    </div>
  {/if}
</div>
