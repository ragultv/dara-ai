<script lang="ts">
  import { onMount } from "svelte";
  import { serverStore } from "./lib/stores/server.svelte.ts";

  onMount(() => {
    void serverStore.init();
    return () => {
      serverStore.stopHealthPolling();
    };
  });
</script>

<main class="min-h-screen flex flex-col items-center justify-center p-6 bg-bg text-text">
  {#if serverStore.isStarting}
    <!-- Splash Screen: Starting the engine -->
    <div
      role="status"
      aria-live="polite"
      class="flex flex-col items-center text-center space-y-6 max-w-sm"
    >
      <!-- Dara geometric mark -->
      <div class="relative flex items-center justify-center w-20 h-20">
        <svg
          class="w-16 h-16"
          viewBox="0 0 64 64"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          aria-hidden="true"
        >
          <!-- Geometric Dara D mark -->
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
  {:else if serverStore.isReady}
    <!-- Ready View: GET /health and loaded model info -->
    <div
      role="region"
      aria-label="Engine status"
      class="w-full max-w-md p-6 rounded-2xl border border-border bg-surface shadow-sm space-y-6"
    >
      <div class="flex items-center justify-between pb-4 border-b border-border">
        <div class="flex items-center space-x-3">
          <svg
            class="w-8 h-8"
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
          <div>
            <h1 class="text-xl font-bold tracking-tight">Dara</h1>
            <p class="text-xs text-text-muted">Engine is ready</p>
          </div>
        </div>

        <span
          class="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-success-subtle text-success"
        >
          ● Active
        </span>
      </div>

      <!-- Engine details -->
      <div class="space-y-4">
        <div>
          <span class="text-xs font-medium uppercase tracking-wider text-text-muted"
            >Loaded Model</span
          >
          <p class="text-lg font-semibold mt-0.5 text-text">
            {serverStore.loadedModel ?? "No model loaded"}
          </p>
        </div>

        <div class="grid grid-cols-2 gap-4 pt-2">
          <div class="p-3 rounded-lg bg-surface-sunken border border-border">
            <span class="text-xs text-text-muted">Status</span>
            <p class="text-sm font-medium mt-0.5 font-mono capitalize">
              {serverStore.health?.status ?? "unknown"}
            </p>
          </div>

          <div class="p-3 rounded-lg bg-surface-sunken border border-border">
            <span class="text-xs text-text-muted">Endpoint</span>
            <p class="text-sm font-medium mt-0.5 font-mono truncate">
              {serverStore.baseUrl}
            </p>
          </div>
        </div>

        <div class="grid grid-cols-2 gap-4">
          <div class="p-3 rounded-lg bg-surface-sunken border border-border">
            <span class="text-xs text-text-muted">Busy</span>
            <p class="text-sm font-medium mt-0.5 font-mono">
              {serverStore.health?.busy ? "Yes" : "No"}
            </p>
          </div>

          <div class="p-3 rounded-lg bg-surface-sunken border border-border">
            <span class="text-xs text-text-muted">Queued Requests</span>
            <p class="text-sm font-medium mt-0.5 font-mono">
              {serverStore.health?.queued ?? 0}
            </p>
          </div>
        </div>
      </div>
    </div>
  {:else}
    <!-- Error or Offline State -->
    <div
      role="alert"
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

      <button
        type="button"
        onclick={() => serverStore.retry()}
        class="inline-flex items-center justify-center px-4 py-2.5 rounded-xl font-medium text-sm bg-primary text-on-primary hover:bg-primary-hover active:bg-primary-active transition-colors cursor-pointer"
      >
        Try again
      </button>
    </div>
  {/if}
</main>
