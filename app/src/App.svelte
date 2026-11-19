<script lang="ts">
  import { onMount } from "svelte";
  import { serverStore } from "./lib/stores/server.svelte.ts";
  import { modelsStore } from "./lib/stores/models.svelte.ts";
  import { router } from "./lib/router/router.svelte.ts";
  import LeftRail from "./lib/shell/LeftRail.svelte";
  import StatusPill from "./lib/shell/StatusPill.svelte";
  import Toast from "./lib/ui/Toast.svelte";
  import Button from "./lib/ui/Button.svelte";
  import ChatPage from "./routes/ChatPage.svelte";
  import ModelsPage from "./routes/ModelsPage.svelte";
  import MonitorPage from "./routes/MonitorPage.svelte";
  import SettingsPage from "./routes/SettingsPage.svelte";
  import KitPage from "./routes/KitPage.svelte";
  import Onboarding from "./routes/Onboarding.svelte";
  import OptionsPage from "./routes/OptionsPage.svelte";

  onMount(() => {
    void serverStore.init();
    void modelsStore.fetchModels();
    return () => {
      serverStore.stopHealthPolling();
    };
  });
</script>

<div class="h-screen w-screen flex flex-col bg-bg text-text overflow-hidden select-none">
  {#if router.current !== "kit" && serverStore.isStarting}
    <!-- Splash Screen: Starting the engine -->
    <main
      class="flex-1 flex flex-col items-center justify-center p-6 bg-bg text-text"
      role="status"
      aria-live="polite"
    >
      <div class="flex flex-col items-center text-center space-y-6 max-w-sm">
        <!-- Dara geometric mark -->
        <div class="relative flex items-center justify-center w-20 h-20">
          <svg
            class="w-16 h-16"
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
        </div>

        <div class="space-y-2">
          <h1 class="text-3xl font-bold tracking-tight">Dara</h1>
          <p class="text-base text-text-muted">Starting the engine</p>
        </div>

        <!-- Pulse indicator -->
        <div class="flex items-center space-x-2" aria-hidden="true">
          <span class="w-2.5 h-2.5 rounded-full bg-primary animate-pulse"></span>
          <span class="text-xs text-text-subtle font-mono">127.0.0.1</span>
        </div>
      </div>
    </main>
  {:else if router.current !== "kit" && (serverStore.status === "error" || serverStore.status === "offline")}
    <!-- Connection Error or Offline View -->
    <main class="flex-1 flex flex-col items-center justify-center p-6 bg-bg text-text" role="alert">
      <div
        class="w-full max-w-md p-6 rounded-2xl border border-border bg-surface shadow-sm space-y-6 text-center"
      >
        <div
          class="inline-flex items-center justify-center w-12 h-12 rounded-full bg-danger-subtle text-danger mx-auto"
        >
          <svg
            class="w-6 h-6"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            stroke-width="2"
            aria-hidden="true"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
            />
          </svg>
        </div>

        <div class="space-y-2">
          <h2 class="text-xl font-bold">Engine Connection Failed</h2>
          <p class="text-sm text-text-muted">
            {serverStore.errorMessage ?? "Could not connect to the engine process."}
          </p>
        </div>

        <Button variant="primary" onclick={() => serverStore.retry()}>Try again</Button>
      </div>
    </main>
  {:else}
    <!-- Main App Shell -->
    <div class="flex-1 flex overflow-hidden">
      <!-- Left Rail Navigation -->
      <LeftRail />

      <!-- Route Workspace Area -->
      <main class="flex-1 flex flex-col overflow-y-auto bg-bg" id="main-content">
        {#if router.current === "chat"}
          {#if !modelsStore.hasInstalledModels && modelsStore.models.length > 0}
            <Onboarding />
          {:else}
            <ChatPage />
          {/if}
        {:else if router.current === "models"}
          <ModelsPage />
        {:else if router.current === "options"}
          <OptionsPage />
        {:else if router.current === "monitor"}
          <MonitorPage />
        {:else if router.current === "settings"}
          <SettingsPage />
        {:else if router.current === "kit"}
          <KitPage />
        {/if}
      </main>
    </div>

    <!-- Status Pill Footer Bar -->
    <StatusPill />
  {/if}

  <!-- Toast Notification Layer -->
  <Toast />
</div>
