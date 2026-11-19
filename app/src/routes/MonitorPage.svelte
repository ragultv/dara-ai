<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { apiClient } from "../lib/api/client.ts";
  import type { Stats, VersionInfo } from "../lib/api/types.ts";
  import Card from "../lib/ui/Card.svelte";
  import Badge from "../lib/ui/Badge.svelte";
  import Button from "../lib/ui/Button.svelte";
  import Meter from "../lib/ui/Meter.svelte";
  import Sparkline from "../lib/ui/Sparkline.svelte";
  import { toast } from "../lib/ui/Toast.svelte";
  import { formatDiagnostics } from "../lib/utils/redact.ts";

  let stats = $state<Stats | null>(null);
  let version = $state<VersionInfo | null>(null);
  let isLoading = $state<boolean>(true);
  let error = $state<string | null>(null);

  // In-memory ring buffer of tok/s samples (max 120)
  let tpsHistory = $state<number[]>([18.2, 22.4, 25.1, 26.5, 26.2, 25.9, 26.8]);
  let pollInterval: ReturnType<typeof setInterval> | null = null;

  async function fetchTelemetry() {
    try {
      const data = await apiClient.getStats();
      stats = data;
      error = null;

      // Update TPS history buffer
      const currentTps =
        data.engine?.live_tokens_per_second > 0
          ? data.engine.live_tokens_per_second
          : data.engine?.avg_decode_tokens_per_second > 0
            ? data.engine.avg_decode_tokens_per_second
            : null;

      if (currentTps !== null) {
        tpsHistory = [...tpsHistory.slice(-119), currentTps];
      }
    } catch (err: unknown) {
      if (!stats) {
        error = err instanceof Error ? err.message : "Failed to load hardware statistics";
      }
    } finally {
      isLoading = false;
    }
  }

  async function fetchVersion() {
    try {
      version = await apiClient.getVersion();
    } catch {
      // optional
    }
  }

  function startPolling() {
    stopPolling();
    void fetchTelemetry();
    pollInterval = setInterval(() => {
      // Only poll when document is visible to keep idle CPU near zero
      if (typeof document !== "undefined" && document.visibilityState === "visible") {
        void fetchTelemetry();
      }
    }, 1000);
  }

  function stopPolling() {
    if (pollInterval) {
      clearInterval(pollInterval);
      pollInterval = null;
    }
  }

  function handleVisibilityChange() {
    if (typeof document !== "undefined") {
      if (document.visibilityState === "visible") {
        void fetchTelemetry();
      }
    }
  }

  onMount(() => {
    void fetchVersion();
    startPolling();

    if (typeof document !== "undefined") {
      document.addEventListener("visibilitychange", handleVisibilityChange);
    }
  });

  onDestroy(() => {
    stopPolling();
    if (typeof document !== "undefined") {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    }
  });

  async function copyDiagnostics() {
    if (!stats) return;
    try {
      const text = formatDiagnostics(stats, version);
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
      }
    } catch {
      // Fallback if clipboard permissions are restricted in certain test environments
    }
    toast.show("Diagnostics copied (usernames redacted)", "success");
  }
</script>

<div class="p-8 max-w-5xl mx-auto space-y-6 text-left">
  <!-- Header with Diagnostics Export -->
  <header
    class="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-border pb-6"
  >
    <div class="space-y-1">
      <div class="flex items-center space-x-2">
        <h1 class="text-2xl font-bold tracking-tight text-text">Hardware & Engine Monitor</h1>
        <Badge variant="success" size="sm">
          <span class="inline-block w-1.5 h-1.5 rounded-full bg-success mr-1.5 animate-pulse"
          ></span>
          Live
        </Badge>
      </div>
      <p class="text-xs text-text-muted">
        Real-time telemetry from your local GPU, VRAM allocation, and generation performance.
      </p>
    </div>

    <Button variant="secondary" size="sm" onclick={copyDiagnostics} disabled={!stats}>
      <svg
        class="w-4 h-4 mr-1.5"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        aria-hidden="true"
      >
        <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
      </svg>
      Copy diagnostics
    </Button>
  </header>

  {#if isLoading && !stats}
    <div class="p-12 text-center text-text-muted space-y-3">
      <div
        class="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto"
      ></div>
      <p class="text-xs">Connecting to engine telemetry...</p>
    </div>
  {:else if error && !stats}
    <div
      role="alert"
      class="p-6 rounded-xl bg-danger-subtle border border-danger text-sm text-danger space-y-3"
    >
      <p class="font-bold">Telemetry Unavailable</p>
      <p class="text-xs">{error}</p>
      <Button variant="secondary" size="sm" onclick={fetchTelemetry}>Retry Connection</Button>
    </div>
  {:else if stats}
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
      <!-- 1. GPU Telemetry Card -->
      <Card variant="surface" padding="md" class="space-y-4">
        <div class="flex items-center justify-between">
          <div class="flex items-center space-x-2">
            <h2 class="text-sm font-bold text-text uppercase tracking-wider">GPU Telemetry</h2>
            <Badge variant="purple" size="sm">{stats.gpu?.name ?? "Integrated GPU"}</Badge>
          </div>
          {#if stats.gpu?.temperature_c}
            <span class="text-xs font-mono font-bold text-text">{stats.gpu.temperature_c}°C</span>
          {/if}
        </div>

        {#if stats.gpu}
          <Meter
            label="VRAM Allocation"
            used={Math.round(stats.gpu.vram_used_mb)}
            capacity={Math.round(stats.gpu.vram_total_mb)}
            unit="MB"
          />

          <div class="grid grid-cols-3 gap-3 pt-2 text-xs font-mono">
            <div class="p-2.5 rounded-lg bg-surface-sunken border border-border">
              <span class="text-text-subtle text-[11px] block">Compute Util</span>
              <p class="text-sm font-bold text-text mt-0.5">{stats.gpu.utilization_percent}%</p>
            </div>
            <div class="p-2.5 rounded-lg bg-surface-sunken border border-border">
              <span class="text-text-subtle text-[11px] block">Power Draw</span>
              <p class="text-sm font-bold text-text mt-0.5">
                {stats.gpu.power_w != null ? `${stats.gpu.power_w.toFixed(1)} W` : "N/A"}
              </p>
            </div>
            <div class="p-2.5 rounded-lg bg-surface-sunken border border-border">
              <span class="text-text-subtle text-[11px] block">SM Clock</span>
              <p class="text-sm font-bold text-text mt-0.5">{stats.gpu.clock_sm_mhz} MHz</p>
            </div>
          </div>
        {/if}
      </Card>

      <!-- 2. Generation Speed & Engine Card -->
      <Card variant="surface" padding="md" class="space-y-4">
        <div class="flex items-center justify-between">
          <h2 class="text-sm font-bold text-text uppercase tracking-wider">Inference Speed</h2>
          <div class="flex items-center space-x-2">
            <span class="text-xs text-text-subtle">Session Speed:</span>
            <Sparkline data={tpsHistory} label="Live generation TPS sparkline" />
          </div>
        </div>

        <div class="grid grid-cols-3 gap-3 pt-2 text-xs font-mono">
          <div class="p-2.5 rounded-lg bg-surface-sunken border border-border">
            <span class="text-text-subtle text-[11px] block">Live Decode</span>
            <p class="text-sm font-bold text-primary mt-0.5">
              {stats.engine.live_tokens_per_second > 0
                ? stats.engine.live_tokens_per_second.toFixed(1)
                : stats.engine.avg_decode_tokens_per_second.toFixed(1)}
              <span class="text-[10px] font-sans text-text-muted">tok/s</span>
            </p>
          </div>
          <div class="p-2.5 rounded-lg bg-surface-sunken border border-border">
            <span class="text-text-subtle text-[11px] block">Avg Prefill</span>
            <p class="text-sm font-bold text-text mt-0.5">
              {stats.engine.avg_prefill_tokens_per_second.toFixed(1)}
              <span class="text-[10px] font-sans text-text-muted">tok/s</span>
            </p>
          </div>
          <div class="p-2.5 rounded-lg bg-surface-sunken border border-border">
            <span class="text-text-subtle text-[11px] block">Completed</span>
            <p class="text-sm font-bold text-text mt-0.5">{stats.engine.requests} reqs</p>
          </div>
        </div>

        <div
          class="flex items-center justify-between text-xs text-text-subtle pt-1 border-t border-border"
        >
          <span>Engine Status: {stats.engine.busy ? "Generating (Busy)" : "Ready (Idle)"}</span>
          <span>Errors: {stats.engine.errors}</span>
        </div>
      </Card>

      <!-- 3. Loaded Model Hardware Placement -->
      <Card variant="surface" padding="md" class="space-y-4">
        <div class="flex items-center justify-between">
          <h2 class="text-sm font-bold text-text uppercase tracking-wider">Model Placement</h2>
          {#if stats.model}
            <Badge variant="purple" size="sm">{stats.model.id}</Badge>
          {:else}
            <Badge variant="neutral" size="sm">No model loaded</Badge>
          {/if}
        </div>

        {#if stats.model}
          <Meter
            label="Context Utilization"
            used={stats.context.held_tokens}
            capacity={stats.context.capacity}
            unit="tokens"
          />

          <div class="grid grid-cols-2 gap-3 pt-2 text-xs font-mono">
            <div class="p-2.5 rounded-lg bg-surface-sunken border border-border">
              <span class="text-text-subtle text-[11px] block">GPU Layers</span>
              <p class="text-sm font-bold text-text mt-0.5">
                {stats.model.layers_on_gpu} / {stats.model.layers} on GPU
              </p>
            </div>
            <div class="p-2.5 rounded-lg bg-surface-sunken border border-border">
              <span class="text-text-subtle text-[11px] block">K/V Precision</span>
              <p class="text-sm font-bold text-text mt-0.5">
                {stats.model.kv_format.toUpperCase()}
              </p>
            </div>
          </div>

          <div class="text-[11px] text-text-subtle space-y-1">
            <p>Placement: <span class="font-mono text-text">{stats.model.placement}</span></p>
            <p>
              Load time: <span class="font-mono text-text"
                >{stats.model.load_seconds.toFixed(2)}s</span
              >
            </p>
          </div>
        {:else}
          <div class="p-6 text-center text-xs text-text-muted">
            Load a model from the Models tab to view its hardware footprint.
          </div>
        {/if}
      </Card>

      <!-- 4. Storage & System Resources -->
      <Card variant="surface" padding="md" class="space-y-4">
        <h2 class="text-sm font-bold text-text uppercase tracking-wider">Storage & System</h2>

        <Meter
          label="Volume Free Space"
          used={Math.round(stats.disk.volume_total_gb - stats.disk.volume_free_gb)}
          capacity={Math.round(stats.disk.volume_total_gb)}
          unit="GB"
        />

        <div class="grid grid-cols-2 gap-3 pt-2 text-xs font-mono">
          <div class="p-2.5 rounded-lg bg-surface-sunken border border-border">
            <span class="text-text-subtle text-[11px] block">Models Stored</span>
            <p class="text-sm font-bold text-text mt-0.5">
              {(stats.disk.installed_models_mb / 1024).toFixed(1)} GB
            </p>
          </div>
          {#if stats.process && stats.process.working_set_mb != null}
            <div class="p-2.5 rounded-lg bg-surface-sunken border border-border">
              <span class="text-text-subtle text-[11px] block">Engine RAM</span>
              <p class="text-sm font-bold text-text mt-0.5">
                {stats.process.working_set_mb.toFixed(0)} MB
              </p>
            </div>
          {/if}
        </div>

        <div class="text-[11px] text-text-subtle font-mono truncate pt-1 border-t border-border">
          Directory: {stats.disk.models_dir}
        </div>
      </Card>
    </div>
  {/if}
</div>
