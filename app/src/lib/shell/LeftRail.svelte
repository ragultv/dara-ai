<script lang="ts">
  import { router, type AppRoute } from "../router/router.svelte.ts";
  import { themeStore, type ThemeMode } from "../stores/theme.svelte.ts";

  interface Props {
    isCollapsed?: boolean;
    ontoggle?: () => void;
  }

  let { isCollapsed = $bindable(false), ontoggle }: Props = $props();

  const navItems: Array<{ route: AppRoute; label: string; icon: string }> = [
    {
      route: "chat",
      label: "Chat",
      icon: "M8 10h.01M12 10h.01M16 10h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
    },
    {
      route: "models",
      label: "Models",
      icon: "M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
    },
    {
      route: "monitor",
      label: "Monitor",
      icon: "M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
    },
    {
      route: "settings",
      label: "Settings",
      icon: "M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z M15 12a3 3 0 11-6 0 3 3 0 016 0z"
    }
  ];

  function handleThemeChange(mode: ThemeMode) {
    themeStore.setMode(mode);
  }
</script>

<aside
  class="flex flex-col justify-between border-r border-border bg-bg-subtle transition-all duration-base select-none shrink-0 {isCollapsed
    ? 'w-16'
    : 'w-58'}"
  aria-label="Sidebar Navigation"
>
  <!-- Top brand and navigation links -->
  <div class="flex flex-col p-3 space-y-4">
    <!-- Header with Dara logo & collapse toggle -->
    <div class="flex items-center justify-between px-1.5 py-1">
      <div class="flex items-center space-x-2.5 overflow-hidden">
        <svg
          class="w-7 h-7 shrink-0"
          viewBox="0 0 64 64"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          aria-hidden="true"
        >
          <rect x="8" y="8" width="18" height="48" rx="6" fill="var(--color-primary)" />
          <path
            d="M26 8H38C49.0457 8 58 16.9543 58 28V36C58 47.0457 49.0457 56 38 56H26V8Z"
            fill="var(--color-primary)"
            fill-opacity="0.85"
          />
          <circle cx="36" cy="32" r="8" fill="var(--color-bg)" />
        </svg>

        {#if !isCollapsed}
          <span class="text-lg font-bold tracking-tight text-text truncate">Dara</span>
        {/if}
      </div>

      <button
        type="button"
        aria-label={isCollapsed ? "Expand navigation rail" : "Collapse navigation rail"}
        onclick={() => {
          isCollapsed = !isCollapsed;
          ontoggle?.();
        }}
        class="p-1 rounded-sm text-text-subtle hover:text-text hover:bg-surface-raised cursor-pointer focus-visible:outline-2 focus-visible:outline-focus"
      >
        <svg
          class="w-4 h-4 transition-transform {isCollapsed ? 'rotate-180' : ''}"
          viewBox="0 0 20 20"
          fill="none"
          stroke="currentColor"
          stroke-width="1.5"
          aria-hidden="true"
        >
          <path d="M12 15l-5-5 5-5" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
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
          class="flex items-center space-x-3 px-3 py-2 rounded-md text-sm font-medium transition-colors cursor-pointer focus-visible:outline-2 focus-visible:outline-focus {isActive
            ? 'bg-primary-subtle text-on-primary-subtle font-semibold'
            : 'text-text-muted hover:text-text hover:bg-surface-raised'}"
        >
          <svg
            class="w-5 h-5 shrink-0"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="1.75"
            aria-hidden="true"
          >
            <path stroke-linecap="round" stroke-linejoin="round" d={item.icon} />
          </svg>
          {#if !isCollapsed}
            <span class="truncate">{item.label}</span>
          {/if}
        </button>
      {/each}
    </nav>
  </div>

  <!-- Bottom: Kit shortcut & Theme selector -->
  <div class="flex flex-col p-3 space-y-3 border-t border-border">
    <!-- UI Kit link (hidden/dev shortcut) -->
    <button
      type="button"
      onclick={() => router.navigate("kit")}
      aria-current={router.current === "kit" ? "page" : undefined}
      title={isCollapsed ? "UI Kit" : undefined}
      class="flex items-center space-x-3 px-3 py-1.5 rounded-md text-xs font-medium transition-colors cursor-pointer text-text-subtle hover:text-text hover:bg-surface-raised {router.current ===
      'kit'
        ? 'bg-primary-subtle text-on-primary-subtle'
        : ''}"
    >
      <svg
        class="w-4 h-4 shrink-0"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="1.75"
        aria-hidden="true"
      >
        <path
          stroke-linecap="round"
          stroke-linejoin="round"
          d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"
        />
      </svg>
      {#if !isCollapsed}
        <span class="truncate">Component Kit</span>
      {/if}
    </button>

    <!-- Theme Switcher -->
    {#if !isCollapsed}
      <div class="flex flex-col space-y-1.5 text-left px-1">
        <span class="text-[11px] font-medium text-text-subtle uppercase tracking-wider">Theme</span>
        <div class="flex p-0.5 rounded bg-surface-sunken border border-border">
          <button
            type="button"
            onclick={() => handleThemeChange("system")}
            aria-pressed={themeStore.mode === "system"}
            class="flex-1 py-1 text-center text-xs font-medium rounded cursor-pointer transition-colors {themeStore.mode ===
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
            class="flex-1 py-1 text-center text-xs font-medium rounded cursor-pointer transition-colors {themeStore.mode ===
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
            class="flex-1 py-1 text-center text-xs font-medium rounded cursor-pointer transition-colors {themeStore.mode ===
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
