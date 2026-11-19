<script lang="ts">
  import { serverStore } from "../stores/server.svelte.ts";
  import Badge from "../ui/Badge.svelte";

  interface Props {
    tokensPerSecond?: number;
    contextUsed?: number;
    contextCapacity?: number;
    class?: string;
  }

  const { tokensPerSecond, contextUsed, contextCapacity, class: className = "" }: Props = $props();

  const isGenerating = $derived(tokensPerSecond !== undefined && tokensPerSecond > 0);
  const statusColor = $derived.by(() => {
    switch (serverStore.status) {
      case "ready":
        return "bg-success";
      case "starting":
        return "bg-warning animate-pulse";
      case "error":
      case "offline":
      default:
        return "bg-danger";
    }
  });

  const statusLabel = $derived.by(() => {
    switch (serverStore.status) {
      case "ready":
        return "Engine is ready";
      case "starting":
        return "Starting Engine";
      case "error":
        return "Engine Error";
      case "offline":
      default:
        return "Engine Offline";
    }
  });
</script>

<footer
  class="h-9 px-4 border-t border-border bg-bg-subtle flex items-center justify-between text-xs text-text-muted select-none shrink-0 {className}"
  role="status"
  aria-label="Application Status"
>
  <!-- Left: Engine Status & Loaded Model -->
  <div class="flex items-center space-x-3">
    <div class="flex items-center space-x-1.5" title={statusLabel}>
      <span class="w-2 h-2 rounded-full {statusColor}"></span>
      <span class="font-medium text-text">{statusLabel}</span>
      {#if serverStore.isReady}
        <span
          class="inline-flex items-center px-1.5 py-0.2 rounded-full text-[10px] font-medium bg-success-subtle text-success"
        >
          ● Active
        </span>
      {/if}
    </div>

    <span class="text-border">|</span>

    <div class="flex items-center space-x-1.5">
      <span class="text-text-subtle">Loaded Model:</span>
      {#if serverStore.loadedModel}
        <Badge variant="purple" size="sm">
          {serverStore.loadedModel}
        </Badge>
      {:else}
        <span class="text-text-muted italic">None</span>
      {/if}
    </div>
  </div>

  <!-- Right: Telemetry, Endpoint & tok/s -->
  <div class="flex items-center space-x-3 font-mono text-[11px]">
    {#if isGenerating}
      <div class="flex items-center space-x-1 text-primary">
        <span class="w-1.5 h-1.5 rounded-full bg-primary animate-ping"></span>
        <span class="tabular-nums font-semibold">{tokensPerSecond?.toFixed(1)} tok/s</span>
      </div>
      <span class="text-border">|</span>
    {/if}

    {#if contextUsed !== undefined && contextCapacity !== undefined}
      <div class="flex items-center space-x-1 text-text-subtle">
        <span>Ctx:</span>
        <span class="tabular-nums">{Math.round((contextUsed / contextCapacity) * 100)}%</span>
      </div>
      <span class="text-border">|</span>
    {/if}

    <div class="text-text-subtle truncate max-w-[220px]">
      <span class="font-sans text-[10px] text-text-muted mr-1">Endpoint</span>
      {serverStore.baseUrl}
    </div>
  </div>
</footer>
