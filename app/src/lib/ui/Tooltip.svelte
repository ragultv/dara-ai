<script lang="ts">
  import type { Snippet } from "svelte";

  interface Props {
    text: string;
    position?: "top" | "bottom" | "left" | "right";
    class?: string;
    children: Snippet;
  }

  const { text, position = "top", class: className = "", children }: Props = $props();

  let isVisible = $state(false);
  const tooltipId = `tooltip-${Math.random().toString(36).substring(2, 9)}`;

  const positionClasses = {
    top: "bottom-full left-1/2 -translate-x-1/2 mb-2",
    bottom: "top-full left-1/2 -translate-x-1/2 mt-2",
    left: "right-full top-1/2 -translate-y-1/2 mr-2",
    right: "left-full top-1/2 -translate-y-1/2 ml-2"
  };
</script>

<div
  class="relative inline-flex {className}"
  role="presentation"
  onmouseenter={() => (isVisible = true)}
  onmouseleave={() => (isVisible = false)}
  onfocusin={() => (isVisible = true)}
  onfocusout={() => (isVisible = false)}
>
  {@render children()}

  {#if isVisible}
    <div
      id={tooltipId}
      role="tooltip"
      class="absolute z-50 px-2.5 py-1 text-xs font-medium rounded-sm bg-surface-raised text-text border border-border shadow-md whitespace-nowrap pointer-events-none {positionClasses[
        position
      ]}"
    >
      {text}
    </div>
  {/if}
</div>
