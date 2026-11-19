<script lang="ts">
  import type { Snippet } from "svelte";

  export interface MenuItem {
    id: string;
    label: string;
    description?: string;
    danger?: boolean;
    disabled?: boolean;
    action: () => void;
  }

  interface Props {
    items: MenuItem[];
    isOpen?: boolean;
    align?: "start" | "end";
    class?: string;
    trigger: Snippet<[{ isOpen: boolean; toggle: () => void }]>;
  }

  let {
    items,
    isOpen = $bindable(false),
    align = "start",
    class: className = "",
    trigger
  }: Props = $props();

  let containerRef: HTMLDivElement | null = $state(null);
  let activeIndex = $state<number>(-1);

  function toggle() {
    isOpen = !isOpen;
    if (isOpen) {
      activeIndex = 0;
    }
  }

  function close() {
    isOpen = false;
    activeIndex = -1;
  }

  function handleKeyDown(e: KeyboardEvent) {
    if (!isOpen) return;

    if (e.key === "Escape") {
      e.stopPropagation();
      close();
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      activeIndex = (activeIndex + 1) % items.length;
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      activeIndex = (activeIndex - 1 + items.length) % items.length;
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      const current = items[activeIndex];
      if (current && !current.disabled) {
        current.action();
        close();
      }
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
</script>

<div bind:this={containerRef} class="relative inline-block {className}">
  {@render trigger({ isOpen, toggle })}

  {#if isOpen}
    <div
      role="menu"
      tabindex="-1"
      class="absolute z-modal mt-1 min-w-[180px] py-1 rounded-md bg-surface-raised border border-border shadow-md {align ===
      'end'
        ? 'right-0'
        : 'left-0'}"
    >
      {#each items as item, i (item.id)}
        <button
          type="button"
          role="menuitem"
          disabled={item.disabled ?? false}
          tabindex={activeIndex === i ? 0 : -1}
          onclick={() => {
            if (!item.disabled) {
              item.action();
              close();
            }
          }}
          class="w-full text-left px-3 py-2 text-xs flex flex-col transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed {activeIndex ===
          i
            ? 'bg-surface-sunken'
            : ''} {item.danger
            ? 'text-danger hover:bg-danger-subtle'
            : 'text-text hover:bg-surface-sunken'}"
        >
          <span class="font-medium">{item.label}</span>
          {#if item.description}
            <span class="text-[10px] text-text-subtle mt-0.5">{item.description}</span>
          {/if}
        </button>
      {/each}
    </div>
  {/if}
</div>
