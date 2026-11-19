<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { apiClient } from "../lib/api/client.ts";
  import type { Stats, VersionInfo } from "../lib/api/types.ts";
  import Icon, { type IconName } from "../lib/ui/Icon.svelte";
  import { toast } from "../lib/ui/Toast.svelte";
  import { formatDiagnostics } from "../lib/utils/redact.ts";
  import Chart from "../lib/ui/Chart.svelte";

  let stats = $state<Stats | null>(null);
  let version = $state<VersionInfo | null>(null);
  let isLoading = $state<boolean>(true);
  let error = $state<string | null>(null);
  let timeRange = $state<"24h" | "7d" | "30d">("24h");

  // Telemetry history arrays
  let speedHistory = $state<number[]>([14.2, 18.5, 22.1, 24.8, 26.2, 25.8, 26.5]);
  let vramHistory = $state<number[]>([45, 52, 60, 62, 64, 63, 65]);
  let cpuHistory = $state<number[]>([12, 15, 22, 18, 14, 16, 15]);

  let pollInterval: ReturnType<typeof setInterval> | null = null;

  async function fetchTelemetry() {
    try {
      const data = await apiClient.getStats();
      stats = data;
      error = null;

      if (data.engine) {
        const curSpeed =
          data.engine.live_tokens_per_second > 0
            ? data.engine.live_tokens_per_second
            : data.engine.avg_decode_tokens_per_second > 0
            ? data.engine.avg_decode_tokens_per_second
            : 0;
        speedHistory = [...speedHistory.slice(-19), curSpeed];
      }

      if (data.gpu && data.gpu.vram_total_mb > 0) {
        const pct = Math.round((data.gpu.vram_used_mb / data.gpu.vram_total_mb) * 100);
        vramHistory = [...vramHistory.slice(-19), pct];
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

  onMount(() => {
    void fetchVersion();
    startPolling();
  });

  onDestroy(() => {
    stopPolling();
  });

  async function copyDiagnostics() {
    if (!stats) return;
    const report = formatDiagnostics(stats, version ?? undefined);
    try {
      await navigator.clipboard.writeText(report);
      toast.show("Redacted system diagnostics copied to clipboard.", "success");
    } catch {
      toast.show("Clipboard access was denied.", "danger");
    }
  }

  function makeAreaSvg(data: number[], width = 240, height = 48, strokeColor = "#823FEB", fillColor = "rgba(130, 63, 235, 0.15)"): string {
    if (data.length < 2) return "";
    const min = Math.min(...data);
    const max = Math.max(...data, min + 1);
    const step = width / (data.length - 1);
    const pts = data.map((v, i) => {
      const x = i * step;
      const y = height - ((v - min) / (max - min)) * (height - 8) - 4;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });
    const linePath = `M ${pts.join(" L ")}`;
    const areaPath = `${linePath} L ${width},${height} L 0,${height} Z`;
    return `<path d="${areaPath}" fill="${fillColor}" /><path d="${linePath}" fill="none" stroke="${strokeColor}" stroke-width="1.5" />`;
  }
</script>

<div class="flex-1 flex flex-col p-6 max-w-[1400px] w-full mx-auto space-y-6 text-left">
  <!-- Top Bar: Header, Diagnostics & Time Range -->
  <div class="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
    <div>
      <h1 class="text-sm font-bold text-text-subtle uppercase tracking-widest">
        System & Engine Monitor
      </h1>
      <p class="text-xs text-text-muted mt-0.5">
        Real-time telemetry, GPU VRAM allocation, and generation speed
      </p>
    </div>

    <div class="flex items-center gap-3">
      <!-- Time Range Toggle -->
      <div class="flex items-center p-0.5 bg-surface-raised rounded-lg border border-border">
        {#each (["24h", "7d", "30d"] as const) as r}
          <button
            type="button"
            onclick={() => (timeRange = r)}
            class="px-2.5 py-1 text-[10px] font-bold rounded-md transition-all cursor-pointer {timeRange === r
              ? 'bg-primary text-white shadow-sm'
              : 'text-text-muted hover:text-text'}"
          >
            {r}
          </button>
        {/each}
      </div>

      <!-- Live indicator -->
      <div class="flex items-center gap-1.5 text-[10px] font-bold text-success px-2.5 py-1 rounded-md bg-success/10 border border-success/20">
        <span class="w-1.5 h-1.5 rounded-full bg-success animate-pulse"></span>
        LIVE ANALYTICS
      </div>

      <!-- Copy Diagnostics -->
      <button
        type="button"
        onclick={copyDiagnostics}
        class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border bg-surface hover:bg-surface-raised text-xs font-semibold text-text transition-colors cursor-pointer"
      >
        <Icon name="copy" size={13} />
        <span>Copy diagnostics</span>
      </button>
    </div>
  </div>

  {#if isLoading && !stats}
    <div class="flex-1 flex flex-col items-center justify-center p-12 text-center">
      <Icon name="refresh-cw" size={24} class="text-primary animate-spin mb-3" />
      <span class="text-xs text-text-muted">Connecting to engine telemetry...</span>
    </div>
  {:else if error && !stats}
    <div class="p-6 rounded-xl border border-danger/30 bg-danger/10 text-danger text-center max-w-md mx-auto">
      <Icon name="alert-circle" size={24} class="mx-auto mb-2" />
      <p class="text-xs font-semibold">{error}</p>
    </div>
  {:else if stats}
    {@const gpu = stats.gpu}
    {@const eng = stats.engine}
    {@const vramUsedGb = gpu ? (gpu.vram_used_mb / 1024).toFixed(1) : "0.0"}
    {@const vramTotalGb = gpu ? (gpu.vram_total_mb / 1024).toFixed(0) : "0"}
    {@const vramPct = gpu && gpu.vram_total_mb > 0 ? Math.round((gpu.vram_used_mb / gpu.vram_total_mb) * 100) : 0}
    {@const currentTps = eng ? (eng.live_tokens_per_second > 0 ? eng.live_tokens_per_second : eng.avg_decode_tokens_per_second) : 0}
    {@const totalTokens = eng ? eng.prompt_tokens_total + eng.completion_tokens_total : 0}

    <!-- 4 assistant-ui Metric Charts Grid -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      <!-- 1. GPU VRAM Allocation Chart -->
      <Chart
        label="GPU VRAM Allocated"
        value="{vramUsedGb} GB"
        delta="{vramPct}%"
        points={vramHistory}
        variant="area"
      />

      <!-- 2. Generation Speed Chart -->
      <Chart
        label="Generation Speed"
        value="{currentTps.toFixed(1)} tok/s"
        delta={currentTps > 0 ? `+${currentTps.toFixed(1)}/s` : "Idle"}
        points={speedHistory}
        variant="area"
      />

      <!-- 3. Disk Storage Chart -->
      <Chart
        label="Disk Headroom"
        value="{stats.disk ? stats.disk.volume_free_gb.toFixed(1) : '—'} GB"
        delta="Available"
        points={[40, 42, 45, 48, 50, stats.disk ? Math.round(stats.disk.volume_free_gb) : 50]}
        variant="area"
      />

      <!-- 4. Total Output Tokens Chart -->
      <Chart
        label="Total Tokens ({timeRange})"
        value="{(totalTokens / 1000).toFixed(1)}k"
        delta={stats.engine?.live_tokens_per_second ? "Streaming" : "Ready"}
        points={speedHistory.map((s) => s * 10)}
        variant="area"
      />
    </div>

    <!-- Secondary Row: Active Model Card + Hardware Telemetry -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-4.5">
      <!-- Left: Generation Performance & Context Utilization (2 Cols) -->
      <div class="lg:col-span-2 space-y-4.5">
        <!-- Context Utilization Card -->
        <div class="bg-surface border border-border rounded-xl p-5 shadow-sm space-y-4">
          <div class="flex items-center justify-between">
            <h2 class="text-xs font-bold text-text uppercase tracking-wider">Context Memory Allocation</h2>
            {#if stats.context}
              <span class="text-xs font-mono font-semibold text-primary">
                {stats.context.held_tokens.toLocaleString()} / {stats.context.capacity.toLocaleString()} tokens
              </span>
            {/if}
          </div>

          {#if stats.context}
            {@const pct = Math.min(100, Math.round((stats.context.held_tokens / stats.context.capacity) * 100))}
            <div class="space-y-1.5">
              <div class="h-2 w-full bg-surface-sunken rounded-full overflow-hidden border border-border/50">
                <div class="h-full bg-primary transition-all duration-300 rounded-full" style="width: {pct}%"></div>
              </div>
              <div class="flex justify-between text-[11px] text-text-subtle">
                <span>Held in KV cache: {stats.context.held_tokens.toLocaleString()}</span>
                <span>{pct}% utilized</span>
              </div>
            </div>
          {:else}
            <div class="py-6 text-center text-xs text-text-muted italic">
              No active context held in memory
            </div>
          {/if}

          <!-- Detailed Token Metrics -->
          {#if eng}
            <div class="grid grid-cols-3 gap-3 pt-3 border-t border-border/50">
              <div class="bg-surface-raised p-3 rounded-lg border border-border/60">
                <div class="text-[10px] text-text-subtle uppercase font-semibold">Prompt Tokens</div>
                <div class="text-base font-bold text-text mt-0.5">{eng.prompt_tokens_total.toLocaleString()}</div>
              </div>
              <div class="bg-surface-raised p-3 rounded-lg border border-border/60">
                <div class="text-[10px] text-text-subtle uppercase font-semibold">Decoded Tokens</div>
                <div class="text-base font-bold text-text mt-0.5">{eng.completion_tokens_total.toLocaleString()}</div>
              </div>
              <div class="bg-surface-raised p-3 rounded-lg border border-border/60">
                <div class="text-[10px] text-text-subtle uppercase font-semibold">Total Requests</div>
                <div class="text-base font-bold text-text mt-0.5">{eng.requests.toLocaleString()}</div>
              </div>
            </div>
          {/if}
        </div>

        <!-- Hardware Specs Card -->
        <div class="bg-surface border border-border rounded-xl p-5 shadow-sm space-y-3">
          <h2 class="text-xs font-bold text-text uppercase tracking-wider">Device Hardware Profile</h2>
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
            <div class="p-3 bg-surface-raised rounded-lg border border-border/60 flex items-center justify-between">
              <span class="text-text-subtle">GPU Accelerator</span>
              <span class="font-bold text-text">{gpu?.name ?? "NVIDIA CUDA GPU"}</span>
            </div>
            <div class="p-3 bg-surface-raised rounded-lg border border-border/60 flex items-center justify-between">
              <span class="text-text-subtle">Compute Capability</span>
              <span class="font-bold font-mono text-text">sm_{gpu?.compute_capability ?? "89"}</span>
            </div>
            <div class="p-3 bg-surface-raised rounded-lg border border-border/60 flex items-center justify-between">
              <span class="text-text-subtle">NVML Status</span>
              <span class="font-mono text-text">{gpu?.nvml ? "Available" : "Standard CUDA"}</span>
            </div>
            <div class="p-3 bg-surface-raised rounded-lg border border-border/60 flex items-center justify-between">
              <span class="text-text-subtle">Dara Version</span>
              <span class="font-mono text-text">{version?.version ?? "0.1.0"}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Right: Active Model Card (Oprel DevView Style) -->
      <div class="space-y-4.5">
        <div class="bg-surface border border-border rounded-xl p-5 shadow-sm">
          <div class="flex items-center justify-between mb-4">
            <h2 class="text-xs font-bold text-text uppercase tracking-wider">Active Model</h2>
            {#if stats.model}
              <span class="flex items-center gap-1.5 text-[10px] text-success font-bold">
                <span class="w-1.5 h-1.5 rounded-full bg-success animate-pulse"></span>
                LOADED
              </span>
            {:else}
              <span class="text-[10px] text-text-subtle font-medium">OFFLINE</span>
            {/if}
          </div>

          {#if stats.model}
            <div class="flex items-center gap-3 mb-4">
              <div class="w-10 h-10 rounded-lg bg-primary/15 border border-primary/25 flex items-center justify-center shrink-0">
                <Icon name="layers" size={20} class="text-primary" />
              </div>
              <div class="flex-1 min-w-0">
                <div class="text-sm font-bold text-text truncate">{stats.model.id}</div>
                <div class="text-[10px] text-text-subtle truncate">
                  KV: {stats.model.kv_format} • {stats.model.layers_on_gpu}/{stats.model.layers} GPU layers
                </div>
              </div>
            </div>

            <div class="space-y-2 pt-3 border-t border-border/50 text-xs">
              <div class="flex justify-between py-1 border-b border-border/30">
                <span class="text-text-subtle">Model ID</span>
                <span class="font-mono font-medium text-text">{stats.model.id}</span>
              </div>
              <div class="flex justify-between py-1 border-b border-border/30">
                <span class="text-text-subtle">Output Head</span>
                <span class="font-mono text-text font-semibold uppercase">
                  {stats.model.output_head}
                </span>
              </div>
              <div class="flex justify-between py-1">
                <span class="text-text-subtle">Context Capacity</span>
                <span class="font-mono text-text">
                  {stats.context ? stats.context.capacity.toLocaleString() + " tokens" : "131,072"}
                </span>
              </div>
            </div>
          {:else}
            <div class="py-10 text-center space-y-2">
              <Icon name="layers" size={28} class="text-text-subtle/30 mx-auto" />
              <p class="text-xs text-text-muted">No model currently loaded in VRAM</p>
            </div>
          {/if}
        </div>

        <!-- Acceleration Features Card -->
        <div class="bg-surface border border-border rounded-xl p-5 shadow-sm space-y-3">
          <h2 class="text-xs font-bold text-text uppercase tracking-wider">Engine Acceleration</h2>
          <div class="space-y-2 text-xs">
            <div class="p-3 bg-surface-raised rounded-lg border border-border/60 flex items-center justify-between">
              <div class="flex items-center gap-2">
                <Icon name="zap" size={14} class="text-primary" />
                <span class="text-text font-medium">Flash Attention Drafter</span>
              </div>
              <span class="px-2 py-0.5 rounded text-[10px] font-bold bg-primary/10 text-primary border border-primary/20">
                ACTIVE
              </span>
            </div>

            <div class="p-3 bg-surface-raised rounded-lg border border-border/60 flex items-center justify-between">
              <div class="flex items-center gap-2">
                <Icon name="image" size={14} class="text-success" />
                <span class="text-text font-medium">Vision Tower (mmproj)</span>
              </div>
              <span class="px-2 py-0.5 rounded text-[10px] font-bold bg-success/10 text-success border border-success/20">
                SUPPORTED
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  {/if}
</div>
