<script lang="ts">
  import { router, type AppRoute } from "../router/router.svelte.ts";
  import { themeStore, type ThemeMode } from "../stores/theme.svelte.ts";
  import Icon, { type IconName } from "../ui/Icon.svelte";

  interface Props {
    isCollapsed?: boolean;
    ontoggle?: () => void;
  }

  let { isCollapsed = $bindable(false), ontoggle }: Props = $props();

  const navItems: Array<{ route: AppRoute; label: string; icon: IconName }> = [
    {
      route: "chat",
      label: "Chat",
      icon: "message-square-plus"
    },
    {
      route: "models",
      label: "Models",
      icon: "box"
    },
    {
      route: "monitor",
      label: "Monitor",
      icon: "activity"
    },
    {
      route: "settings",
      label: "Settings",
      icon: "settings"
    }
  ];

  function handleThemeChange(mode: ThemeMode) {
    themeStore.setMode(mode);
  }
</script>

<aside
  class="flex flex-col justify-between border-r border-border bg-bg-subtle transition-all duration-base select-none shrink-0 {isCollapsed
    ? 'w-16'
    : 'w-56'}"
  aria-label="Sidebar Navigation"
>
  <!-- Top brand and navigation links -->
  <div class="flex flex-col p-3 space-y-4">
    <!-- Header with Dara logo & collapse toggle -->
    <div class="flex items-center justify-between px-1.5 py-1">
      <div class="flex items-center space-x-2.5 overflow-hidden">
        <img
          src="/dara.png"
          alt="Dara logo"
          class="w-7 h-7 shrink-0 object-contain"
        />

        {#if !isCollapsed}
          <div class="flex flex-col text-left">
            <span class="text-sm font-bold tracking-tight text-text leading-tight">DARA STUDIO</span
            >
            <span class="text-[10px] text-text-subtle font-medium">Local GPU Inference</span>
          </div>
        {/if}
      </div>

      <button
        type="button"
        aria-label={isCollapsed ? "Expand navigation rail" : "Collapse navigation rail"}
        onclick={() => {
          isCollapsed = !isCollapsed;
          ontoggle?.();
        }}
        class="p-1 rounded-md text-text-subtle hover:text-text hover:bg-surface-raised cursor-pointer focus-visible:outline-2 focus-visible:outline-focus"
      >
        <Icon name={isCollapsed ? "chevron-right" : "chevron-left"} size={14} />
      </button>
    </div>

    <!-- Nav buttons -->
    <nav class="flex flex-col space-y-1" aria-label="Main">
      {#each navItems as item (item.route)}
        {@const isActive = router.current === item.route}
        <button
          type="button"
          onclick={() => router.navigate(item.route)}
          aria-current={isActive ? "page" : undefined}
          title={isCollapsed ? item.label : undefined}
          class="flex items-center space-x-3 px-3 py-2.5 rounded-lg text-xs font-semibold transition-all cursor-pointer border {isActive
            ? 'bg-surface-raised text-primary border-primary/30 shadow-sm font-bold'
            : 'border-transparent text-text-muted hover:text-text hover:bg-surface-raised/70'}"
        >
          <Icon name={item.icon} size={16} class={isActive ? "text-primary" : "text-text-muted"} />
          {#if !isCollapsed}
            <span class="truncate">{item.label}</span>
          {/if}
        </button>
      {/each}
    </nav>
  </div>

  <!-- Bottom: Theme selector -->
  <div class="flex flex-col p-3 space-y-3 border-t border-border">
    <!-- Theme Switcher -->
    {#if !isCollapsed}
      <div class="flex flex-col space-y-1.5 text-left px-1">
        <span class="text-[10px] font-bold text-text-subtle uppercase tracking-wider">Theme</span>
        <div class="flex p-0.5 rounded-lg bg-surface-sunken border border-border">
          <button
            type="button"
            onclick={() => handleThemeChange("system")}
            aria-pressed={themeStore.mode === "system"}
            class="flex-1 py-1 text-center text-xs font-medium rounded-md cursor-pointer transition-colors {themeStore.mode ===
            'system'
              ? 'bg-surface text-text shadow-sm'
              : 'text-text-muted hover:text-text'}"
          >
            System
          </button>
          <button
            type="button"
            onclick={() => handleThemeChange("light")}
            aria-pressed={themeStore.mode === "light"}
            class="flex-1 py-1 text-center text-xs font-medium rounded-md cursor-pointer transition-colors {themeStore.mode ===
            'light'
              ? 'bg-surface text-text shadow-sm'
              : 'text-text-muted hover:text-text'}"
          >
            Light
          </button>
          <button
            type="button"
            onclick={() => handleThemeChange("dark")}
            aria-pressed={themeStore.mode === "dark"}
            class="flex-1 py-1 text-center text-xs font-medium rounded-md cursor-pointer transition-colors {themeStore.mode ===
            'dark'
              ? 'bg-surface text-text shadow-sm'
              : 'text-text-muted hover:text-text'}"
          >
            Dark
          </button>
        </div>
      </div>
    {/if}
  </div>
</aside>
