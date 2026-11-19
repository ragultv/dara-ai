<script lang="ts">
  import type { Snippet } from "svelte";

  export interface TabItem {
    id: string;
    label: string;
    badge?: string;
  }

  interface Props {
    tabs: TabItem[];
    activeTab?: string;
    class?: string;
    onchange?: (id: string) => void;
    children?: Snippet<[{ activeTab: string }]>;
  }

  let {
    tabs,
    activeTab = $bindable(tabs[0]?.id ?? ""),
    class: className = "",
    onchange,
    children
  }: Props = $props();

  function selectTab(id: string) {
    activeTab = id;
    onchange?.(id);
  }

  function handleKeyDown(e: KeyboardEvent, index: number) {
    if (e.key === "ArrowRight") {
      e.preventDefault();
      const nextIndex = (index + 1) % tabs.length;
      const nextTab = tabs[nextIndex];
      if (nextTab) selectTab(nextTab.id);
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      const prevIndex = (index - 1 + tabs.length) % tabs.length;
      const prevTab = tabs[prevIndex];
      if (prevTab) selectTab(prevTab.id);
    }
  }
</script>

<div class="flex flex-col space-y-4 w-full {className}">
  <!-- Tab Header List -->
  <div
    role="tablist"
    class="flex items-center space-x-1 border-b border-border overflow-x-auto text-sm"
  >
    {#each tabs as tab, i (tab.id)}
      {@const isSelected = activeTab === tab.id}
      <button
        type="button"
        role="tab"
        id="tab-{tab.id}"
        aria-controls="tabpanel-{tab.id}"
        aria-selected={isSelected}
        tabindex={isSelected ? 0 : -1}
        onclick={() => selectTab(tab.id)}
        onkeydown={(e) => handleKeyDown(e, i)}
        class="pb-2.5 px-3 font-medium transition-colors cursor-pointer select-none border-b-2 -mb-px flex items-center space-x-2 focus-visible:outline-2 focus-visible:outline-focus focus-visible:outline-offset-2 {isSelected
          ? 'border-primary text-primary'
          : 'border-transparent text-text-muted hover:text-text hover:border-border'}"
      >
        <span>{tab.label}</span>
        {#if tab.badge}
          <span
            class="px-1.5 py-0.2 rounded-full text-[10px] bg-surface-sunken text-text-subtle font-mono"
            >{tab.badge}</span
          >
        {/if}
      </button>
    {/each}
  </div>

  <!-- Tab Panel Content -->
  {#if children}
    <div
      role="tabpanel"
      id="tabpanel-{activeTab}"
      aria-labelledby="tab-{activeTab}"
      class="outline-none"
    >
      {@render children({ activeTab })}
    </div>
  {/if}
</div>
