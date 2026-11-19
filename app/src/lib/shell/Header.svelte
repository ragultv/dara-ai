<script lang="ts">
  import { router } from "../router/router.svelte.ts";
  import { modelsStore } from "../stores/models.svelte.ts";
  import { serverStore } from "../stores/server.svelte.ts";
  import Icon from "../ui/Icon.svelte";

  let dropdownOpen = $state(false);

  const currentRouteName = $derived.by(() => {
    switch (router.current) {
      case "chat":
        return "Chat";
      case "models":
        return "Models";
      case "monitor":
        return "Monitor";
      case "options":
        return "Model Options";
      case "settings":
        return "Settings";
      default:
        return "Dara Studio";
    }
  });

  const activeModel = $derived.by(() => {
    return modelsStore.loadedModel ?? (serverStore.loadedModel ? modelsStore.models.find(m => m.id === serverStore.loadedModel) : null);
  });

  async function handleSelectModel(modelId: string) {
    dropdownOpen = false;
    if (activeModel?.id === modelId) return;
    await modelsStore.loadModel(modelId);
  }
</script>

<header class="h-12 border-b border-border bg-bg-subtle/80 backdrop-blur-md px-5 flex items-center justify-between shrink-0 select-none z-20">
  <!-- Left: Page Title / Breadcrumb -->
  <div class="flex items-center gap-2.5">
    <span class="text-xs font-bold text-text-muted uppercase tracking-wider">{currentRouteName}</span>
  </div>

  <!-- Right: Model Switcher Dropdown & Live Status -->
  <div class="flex items-center gap-3">
    <!-- Model Switcher Dropdown -->
    <div class="relative">
      <button
        type="button"
        onclick={() => (dropdownOpen = !dropdownOpen)}
        class="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-border bg-surface hover:bg-surface-raised text-xs font-semibold text-text transition-all cursor-pointer shadow-sm"
        aria-expanded={dropdownOpen}
        aria-haspopup="true"
      >
        <Icon name="cpu" size={13} class="text-primary" />
        <span class="max-w-[140px] truncate">
          {activeModel?.dara.name ?? (activeModel?.id ? activeModel.id : "Switch Model")}
        </span>
        {#if activeModel}
          <span class="w-1.5 h-1.5 rounded-full bg-success"></span>
        {/if}
        <Icon name="chevron-down" size={12} class="text-text-subtle transition-transform {dropdownOpen ? 'rotate-180' : ''}" />
      </button>

      {#if dropdownOpen}
        <!-- Backdrop -->
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <!-- svelte-ignore a11y_click_events_have_key_events -->
        <div class="fixed inset-0 z-30" onclick={() => (dropdownOpen = false)}></div>

        <!-- Dropdown Menu -->
        <div class="absolute right-0 top-full mt-1.5 w-76 rounded-xl border border-border bg-surface-raised shadow-xl p-1.5 z-40 text-left animate-in fade-in zoom-in-95 duration-100">
          <div class="px-2.5 py-1.5 text-[10px] font-bold text-text-subtle uppercase tracking-wider border-b border-border/50">
            Available Models
          </div>
          <div class="max-h-60 overflow-y-auto p-1 space-y-0.5">
            {#if modelsStore.models.length === 0}
              <div class="p-3 text-center text-xs text-text-muted italic">No models registered</div>
            {:else}
              {#each modelsStore.models as m (m.id)}
                {@const isLoaded = m.dara.loaded}
                {@const isInstalled = m.dara.status === "installed"}
                <button
                  type="button"
                  onclick={() => handleSelectModel(m.id)}
                  disabled={!isInstalled}
                  class="w-full text-left px-2.5 py-2 rounded-lg text-xs transition-colors flex items-center justify-between gap-2 {isLoaded
                    ? 'bg-primary/10 text-primary font-bold'
                    : isInstalled
                    ? 'text-text hover:bg-surface cursor-pointer'
                    : 'text-text-muted opacity-50 cursor-not-allowed'}"
                >
                  <div class="min-w-0 flex-1">
                    <div class="font-semibold truncate text-[12px]">{m.dara.name}</div>
                    <div class="text-[10px] text-text-subtle flex items-center gap-1.5 mt-0.5">
                      <span>{(m.dara.download_size_mb / 1024).toFixed(1)} GB</span>
                      <span>•</span>
                      <span class="font-mono">{m.dara.quant}</span>
                    </div>
                  </div>
                  {#if isLoaded}
                    <span class="px-1.5 py-0.5 rounded text-[9px] font-bold bg-success/15 text-success border border-success/30">
                      LOADED
                    </span>
                  {:else if isInstalled}
                    <span class="text-[10px] text-primary font-medium hover:underline">
                      Load
                    </span>
                  {:else}
                    <span class="text-[9px] text-text-subtle uppercase">
                      Not installed
                    </span>
                  {/if}
                </button>
              {/each}
            {/if}
          </div>
        </div>
      {/if}
    </div>

    <!-- Live indicator -->
    <div class="flex items-center gap-1.5 text-[10px] text-text-subtle font-mono pl-2 border-l border-border">
      <span class="w-1.5 h-1.5 rounded-full {serverStore.isReady ? 'bg-success animate-pulse' : 'bg-danger'}"></span>
      <span>{serverStore.isReady ? 'ONLINE' : 'OFFLINE'}</span>
    </div>
  </div>
</header>
