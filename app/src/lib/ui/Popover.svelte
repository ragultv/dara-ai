<script lang="ts">
  import type { Snippet } from "svelte";

  interface Props {
    id?: string;
    isOpen?: boolean;
    align?: "start" | "center" | "end";
    class?: string;
    trigger: Snippet<[{ isOpen: boolean; toggle: () => void }]>;
    children: Snippet<[{ close: () => void }]>;
  }

  let {
    id,
    isOpen = $bindable(false),
    align = "start",
    class: className = "",
    trigger,
    children
  }: Props = $props();

  const generatedId = `popover-${Math.random().toString(36).substring(2, 9)}`;
  const popoverId = $derived(id ?? generatedId);
  let containerRef: HTMLDivElement | null = $state(null);

  function toggle() {
    isOpen = !isOpen;
  }

  function close() {
    isOpen = false;
  }

  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === "Escape" && isOpen) {
      e.stopPropagation();
      close();
    }
  }

  function handleWindowClick(e: MouseEvent) {
    if (isOpen && containerRef && !containerRef.contains(e.target as Node)) {
      close();
    }
  }

  $effect(() => {
    if (!isOpen) return;

    window.addEventListener("click", handleWindowClick, true);
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("click", handleWindowClick, true);
      window.removeEventListener("keydown", handleKeyDown);
    };
  });

  const alignClasses = {
    start: "left-0",
    center: "left-1/2 -translate-x-1/2",
    end: "right-0"
  };
</script>

<div bind:this={containerRef} class="relative inline-block {className}">
  {@render trigger({ isOpen, toggle })}

  {#if isOpen}
    <div
      id={popoverId}
      role="region"
      class="absolute z-modal mt-2 min-w-[200px] rounded-lg bg-surface-raised border border-border shadow-lg p-4 {alignClasses[
        align
      ]}"
    >
      {@render children({ close })}
    </div>
  {/if}
</div>
