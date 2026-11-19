<script lang="ts">
  import { onMount } from "svelte";
  import { apiClient } from "../lib/api/client.ts";
  import { router } from "../lib/router/router.svelte.ts";
  import { modelsStore } from "../lib/stores/models.svelte.ts";
  import type { OptionsResponse, KvFormat, OptionEntry, BenchmarkJob } from "../lib/api/types.ts";
  import Icon from "../lib/ui/Icon.svelte";
  import Dialog from "../lib/ui/Dialog.svelte";
  import Button from "../lib/ui/Button.svelte";
  import Toggle from "../lib/ui/Toggle.svelte";
  import { toast } from "../lib/ui/Toast.svelte";

  let optionsData = $state<OptionsResponse | null>(null);
  let isLoading = $state<boolean>(true);
  let error = $state<string | null>(null);

  let selectedKv = $state<KvFormat>("q8");
  let selectedFlash = $state<boolean>(true);
  let selectedContext = $state<number | undefined>(undefined);

  let isBenchmarkModalOpen = $state<boolean>(false);
  let activeBenchmark = $state<BenchmarkJob | null>(null);
  let isBenchmarking = $state<boolean>(false);
  let benchmarkPollInterval: ReturnType<typeof setInterval> | null = null;

  const modelId = $derived(router.optionsModelId || "dara-nano");
  const currentModelMeta = $derived(modelsStore.models.find((m) => m.id === modelId));

  onMount(() => {
    void loadOptions();
    return () => {
      stopBenchmarkPolling();
    };
  });

  async function loadOptions() {
    isLoading = true;
    error = null;
    try {
      const data = await apiClient.getModelOptions(modelId);
      optionsData = data;
      if (data.recommended) {
        selectedKv = data.recommended.kv;
        selectedFlash = data.recommended.flash;
        selectedContext = data.recommended.context;
      } else if (data.formats[0]) {
        selectedKv = data.formats[0].id;
      }
    } catch (err: unknown) {
      error = err instanceof Error ? err.message : "Failed to load model options";
    } finally {
      isLoading = false;
    }
  }

  const currentOption = $derived<OptionEntry | null>(
    optionsData?.options.find(
      (opt) => opt.kv === selectedKv && Boolean(opt.flash) === selectedFlash
    ) ?? null
  );

  const isRecommended = $derived(
    Boolean(
      optionsData?.recommended &&
      optionsData.recommended.kv === selectedKv &&
      Boolean(optionsData.recommended.flash) === selectedFlash
    )
  );

  async function startBenchmark() {
    isBenchmarking = true;
    try {
      const job = await apiClient.startBenchmark(modelId);
      activeBenchmark = job;
      startBenchmarkPolling(job.id);
    } catch (err: unknown) {
      toast.show(err instanceof Error ? err.message : "Benchmark failed to start", "danger");
      isBenchmarking = false;
    }
  }

  function startBenchmarkPolling(jobId: string) {
    stopBenchmarkPolling();
    benchmarkPollInterval = setInterval(async () => {
      try {
        const job = await apiClient.getBenchmarkJob(jobId);
        activeBenchmark = job;
        if (job.status === "done" || job.status === "failed" || job.status === "cancelled") {
          stopBenchmarkPolling();
          isBenchmarking = false;
          if (job.status === "done") {
            toast.show("Benchmark completed! New speed measurements saved.", "success");
            void loadOptions();
          }
        }
      } catch {
        stopBenchmarkPolling();
        isBenchmarking = false;
      }
    }, 1000);
  }

  function stopBenchmarkPolling() {
    if (benchmarkPollInterval) {
      clearInterval(benchmarkPollInterval);
      benchmarkPollInterval = null;
    }
  }

  async function cancelBenchmark() {
    if (activeBenchmark) {
      try {
        await apiClient.cancelBenchmark(activeBenchmark.id);
      } catch {
        // ignore
      }
      stopBenchmarkPolling();
      isBenchmarking = false;
      activeBenchmark = null;
      isBenchmarkModalOpen = false;
      toast.show("Benchmark was cancelled.", "info");
    }
  }

  async function applyAndLoad() {
    if (!optionsData) return;
    try {
      await modelsStore.loadModel(modelId, {
        kv: selectedKv,
        flash: selectedFlash,
        ...(selectedContext !== undefined ? { ctx: selectedContext } : {})
      });
      toast.show(`Loaded ${modelId} with ${selectedKv.toUpperCase()} settings.`, "success");
      router.navigate("chat");
    } catch (err: unknown) {
      toast.show(err instanceof Error ? err.message : "Failed to load settings into VRAM.", "danger");
    }
  }

  function formatContextLabel(num: number): string {
    if (num >= 1000) return `${Math.round(num / 1024)}k`;
    return `${num}`;
  }
</script>

<div class="p-6 max-w-5xl w-full mx-auto space-y-6 text-left">
  <!-- Top Navigation & Back Button -->
  <div class="flex items-center justify-between border-b border-border pb-4">
    <button
      type="button"
      onclick={() => router.navigate("models")}
      class="flex items-center gap-1.5 text-xs font-semibold text-text-muted hover:text-text transition-colors cursor-pointer"
    >
      <Icon name="chevron-left" size={14} />
      <span>Back to Models</span>
    </button>
  </div>

  {#if isLoading}
    <div class="py-16 text-center space-y-3">
      <Icon name="refresh-cw" size={24} class="text-primary animate-spin mx-auto" />
      <p class="text-xs text-text-muted">Analyzing GPU placement options for {modelId}...</p>
    </div>
  {:else if error}
    {#if error.toLowerCase().includes("not installed")}
      <!-- Not Installed Friendly State -->
      <div class="max-w-md mx-auto text-center space-y-4 bg-surface border border-border rounded-xl p-8 shadow-sm">
        <div class="w-12 h-12 rounded-xl bg-primary/10 border border-primary/20 text-primary flex items-center justify-center mx-auto">
          <Icon name="download" size={22} />
        </div>
        <div>
          <h2 class="text-base font-bold text-text">Model Not Downloaded Yet</h2>
          <p class="text-xs text-text-muted mt-1 leading-relaxed">
            Model <strong class="text-text font-mono">{modelId}</strong> must be downloaded before you
            can configure GPU placement, K/V cache precision, or run benchmark sweeps.
          </p>
        </div>
        <div class="pt-2">
          <button
            type="button"
            onclick={() => {
              void modelsStore.downloadModel(modelId);
              router.navigate("models");
            }}
            class="h-9 px-4 rounded-lg bg-primary hover:bg-primary-hover text-white text-xs font-semibold transition-all cursor-pointer shadow-sm"
          >
            Download {currentModelMeta ? `(${(currentModelMeta.dara.download_size_mb / 1024).toFixed(1)} GB)` : "Model"}
          </button>
        </div>
      </div>
    {:else}
      <div role="alert" class="p-6 rounded-xl bg-danger/10 border border-danger/30 text-xs text-danger text-center max-w-md mx-auto space-y-3">
        <Icon name="alert-circle" size={24} class="mx-auto" />
        <p class="font-bold">Failed to load model options</p>
        <p class="text-text-subtle">{error}</p>
        <button
          type="button"
          onclick={loadOptions}
          class="h-8 px-4 rounded-lg bg-surface border border-border text-xs font-semibold text-text hover:bg-surface-raised transition-colors cursor-pointer"
        >
          Try again
        </button>
      </div>
    {/if}
  {:else if optionsData}
    <!-- Options Main View -->
    <header class="space-y-4 border-b border-border pb-6">
      <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div class="flex items-center gap-2">
            <h1 class="text-lg font-bold text-text tracking-tight">{optionsData.name} Options</h1>
            <span class="px-2 py-0.5 rounded-md bg-primary/10 border border-primary/20 text-primary text-[10px] font-bold">
              {optionsData.family}
            </span>
          </div>
          <p class="text-xs text-text-muted mt-1">
            Configure quantization precision and speculative drafting tailored for your graphics hardware.
          </p>
        </div>

        <!-- GPU Hardware Specs Card -->
        <div class="flex items-center gap-3 bg-surface p-3 rounded-xl border border-border text-xs font-mono shadow-sm">
          <div class="w-8 h-8 rounded-lg bg-primary/10 border border-primary/20 text-primary flex items-center justify-center shrink-0">
            <Icon name="cpu" size={16} />
          </div>
          <div>
            <div class="font-bold text-text truncate max-w-[200px]">{optionsData.machine.gpu}</div>
            <div class="text-[10px] text-text-subtle">
              VRAM: {(optionsData.machine.vram_free_assumed_mb / 1024).toFixed(1)} GB free of {(optionsData.machine.vram_total_mb / 1024).toFixed(0)} GB
            </div>
          </div>
        </div>
      </div>
    </header>

    <!-- Controls Grid: K/V Cache Precision & Flash Drafter -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-5">
      <!-- K/V Precision Cards (2 cols) -->
      <div class="md:col-span-2 space-y-3">
        <h2 class="text-xs font-bold text-text-subtle uppercase tracking-wider">K/V Cache Precision</h2>

        <div class="grid grid-cols-1 sm:grid-cols-2 gap-3" role="radiogroup" aria-label="K/V precision options">
          {#each optionsData.formats as fmt (fmt.id)}
            {@const isSelected = selectedKv === fmt.id}
            <button
              type="button"
              role="radio"
              aria-checked={isSelected}
              onclick={() => (selectedKv = fmt.id)}
              class="p-4 rounded-xl border text-left transition-all cursor-pointer space-y-2 {isSelected
                ? 'bg-surface-raised border-primary ring-1 ring-primary/40 shadow-sm'
                : 'bg-surface border-border hover:border-primary/30'}"
            >
              <div class="flex items-center justify-between">
                <span class="font-bold text-xs text-text">{fmt.name}</span>
                <span class="px-1.5 py-0.5 rounded text-[9px] font-bold bg-surface-sunken border border-border text-text-subtle">
                  {fmt.quality.text}
                </span>
              </div>
              <p class="text-[11px] text-text-muted leading-relaxed">{fmt.description}</p>
              {#if fmt.caveat}
                <p class="text-[10px] text-warning italic leading-snug">{fmt.caveat}</p>
              {/if}
            </button>
          {/each}
        </div>
      </div>

      <!-- Flash Acceleration Card -->
      <div class="space-y-3">
        <h2 class="text-xs font-bold text-text-subtle uppercase tracking-wider">Flash Acceleration</h2>
        <div class="bg-surface border border-border rounded-xl p-4 space-y-3 shadow-sm">
          <Toggle
            label="Enable Flash Drafter"
            description="Speculative draft decoding speeds up inference up to 2-3x."
            disabled={!optionsData.flash ||
              typeof optionsData.flash !== "object" ||
              !("available" in optionsData.flash) ||
              !optionsData.flash.available}
            bind:checked={selectedFlash}
          />
        </div>
      </div>
    </div>

    <!-- Active Profile Performance Summary -->
    {#if currentOption}
      <div class="bg-surface border border-border rounded-xl p-5 shadow-sm space-y-5">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div class="flex items-center gap-2">
              <h3 class="text-sm font-bold text-text">
                Fastest Setup: {formatContextLabel(currentOption.fast.context)} context
              </h3>
              {#if isRecommended}
                <span class="px-2 py-0.5 rounded-md text-[10px] font-bold bg-success/10 text-success border border-success/20">
                  Recommended
                </span>
              {/if}
            </div>
            {#if isRecommended && optionsData.recommended.reason}
              <p class="text-xs text-success font-medium mt-0.5">{optionsData.recommended.reason}</p>
            {/if}
          </div>

          <!-- Speed Box -->
          <div class="text-right">
            {#if currentOption.fast.speed.source === "measured"}
              <div class="text-2xl font-bold font-mono text-primary tabular-nums">
                {currentOption.fast.speed.decode_tps_start.toFixed(1)}
                <span class="text-xs font-sans text-text-muted font-normal">tok/s</span>
              </div>
            {:else}
              <div class="text-xs font-mono text-text-muted italic">Not benchmarked</div>
              <button
                type="button"
                onclick={() => (isBenchmarkModalOpen = true)}
                class="text-xs text-primary hover:underline cursor-pointer"
              >
                Benchmark on this GPU
              </button>
            {/if}
          </div>
        </div>

        <!-- Context Ladder Table -->
        <div class="space-y-2">
          <span class="text-[10px] font-bold text-text-subtle uppercase tracking-wider">Context Ladder</span>
          <div class="overflow-x-auto border border-border rounded-lg bg-surface-sunken">
            <table class="w-full text-xs text-left">
              <thead class="border-b border-border text-text-subtle uppercase text-[10px] font-mono">
                <tr>
                  <th class="p-2.5">Context</th>
                  <th class="p-2.5">GPU Placement</th>
                  <th class="p-2.5">Cache VRAM</th>
                  <th class="p-2.5 text-right">Generation Speed</th>
                </tr>
              </thead>
              <tbody class="divide-y divide-border">
                {#each currentOption.points as pt (pt.context)}
                  <tr class="hover:bg-surface-raised/50 transition-colors">
                    <td class="p-2.5 font-bold font-mono text-text">{formatContextLabel(pt.context)}</td>
                    <td class="p-2.5 text-text-subtle">
                      {pt.fully_on_gpu ? "Full GPU" : `${pt.layers_gpu}/${pt.layers} GPU layers`}
                    </td>
                    <td class="p-2.5 font-mono text-text-subtle">{(pt.kv_cache_mb / 1024).toFixed(2)} GB</td>
                    <td class="p-2.5 text-right font-mono font-semibold text-text">
                      {#if pt.speed.source === "measured"}
                        <span class="text-success">{pt.speed.decode_tps_start.toFixed(1)} tok/s</span>
                      {:else}
                        <span class="text-text-muted italic">Not measured</span>
                      {/if}
                    </td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        </div>

        <!-- Action Buttons -->
        <div class="flex items-center justify-between pt-4 border-t border-border/50">
          <button
            type="button"
            onclick={() => (isBenchmarkModalOpen = true)}
            class="h-9 px-4 rounded-lg border border-border bg-surface hover:bg-surface-raised text-xs font-semibold text-text transition-colors cursor-pointer flex items-center gap-2"
          >
            <Icon name="activity" size={13} />
            <span>Run Benchmark</span>
          </button>

          <button
            type="button"
            onclick={applyAndLoad}
            class="h-9 px-5 rounded-lg bg-primary hover:bg-primary-hover text-white text-xs font-semibold transition-all cursor-pointer shadow-sm flex items-center gap-2"
          >
            <Icon name="check" size={13} />
            <span>Apply & Load into VRAM</span>
          </button>
        </div>
      </div>
    {/if}
  {/if}

  <!-- Benchmark Dialog -->
  <Dialog
    isOpen={isBenchmarkModalOpen}
    title="GPU Inference Benchmark"
    onclose={() => {
      if (!isBenchmarking) isBenchmarkModalOpen = false;
    }}
  >
    <div class="space-y-4 text-xs text-text-muted">
      <p>
        The benchmark evaluates actual token generation speeds across K/V precision levels and Flash drafter configs on your GPU.
      </p>

      {#if activeBenchmark}
        <div class="p-3 bg-surface-sunken border border-border rounded-xl space-y-2">
          <div class="flex justify-between items-center text-xs">
            <span class="font-bold text-text">Status: {activeBenchmark.status}</span>
          </div>
          {#if activeBenchmark.log && activeBenchmark.log.length > 0}
            <div class="max-h-32 overflow-y-auto font-mono text-[10px] text-text-subtle p-2 bg-surface rounded">
              {#each activeBenchmark.log as line, i (i)}
                <div>{line}</div>
              {/each}
            </div>
          {/if}
        </div>
      {/if}
    </div>

    {#snippet actions()}
      {#if isBenchmarking}
        <Button variant="danger" size="sm" onclick={cancelBenchmark}>Cancel</Button>
      {:else}
        <Button variant="ghost" size="sm" onclick={() => (isBenchmarkModalOpen = false)}>Close</Button>
        <Button variant="primary" size="sm" onclick={startBenchmark}>Start Benchmark</Button>
      {/if}
    {/snippet}
  </Dialog>
</div>
