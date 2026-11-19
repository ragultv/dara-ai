<script lang="ts">
  import { onMount } from "svelte";
  import { apiClient } from "../lib/api/client.ts";
  import { router } from "../lib/router/router.svelte.ts";
  import { modelsStore } from "../lib/stores/models.svelte.ts";
  import type { OptionsResponse, KvFormat, OptionEntry, BenchmarkJob } from "../lib/api/types.ts";
  import Card from "../lib/ui/Card.svelte";
  import Badge from "../lib/ui/Badge.svelte";
  import Button from "../lib/ui/Button.svelte";
  import Toggle from "../lib/ui/Toggle.svelte";
  import Dialog from "../lib/ui/Dialog.svelte";
  import ProgressBar from "../lib/ui/ProgressBar.svelte";
  import { toast } from "../lib/ui/Toast.svelte";

  let optionsData = $state<OptionsResponse | null>(null);
  let isLoading = $state<boolean>(true);
  let error = $state<string | null>(null);

  // Selected format and flash
  let selectedKv = $state<KvFormat>("q8");
  let selectedFlash = $state<boolean>(true);
  let selectedContext = $state<number | undefined>(undefined);

  // Benchmark state
  let isBenchmarkModalOpen = $state<boolean>(false);
  let activeBenchmark = $state<BenchmarkJob | null>(null);
  let isBenchmarking = $state<boolean>(false);
  let benchmarkPollInterval: ReturnType<typeof setInterval> | null = null;

  const modelId = $derived(router.optionsModelId || "dara-nano");

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
      // Initialize with recommended values
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

  // Active option entry matching chosen (kv, flash)
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
            toast.show("Benchmark completed!", "success");
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
      toast.show("Benchmark cancelled", "info");
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
      toast.show(`Loaded ${modelId} with ${selectedKv.toUpperCase()} settings`, "success");
      router.navigate("chat");
    } catch (err: unknown) {
      toast.show(err instanceof Error ? err.message : "Failed to load settings", "danger");
    }
  }

  function formatContextLabel(num: number): string {
    if (num >= 1000) return `${Math.round(num / 1024)}K`;
    return `${num}`;
  }
</script>

<div class="p-8 max-w-5xl mx-auto space-y-8 text-left">
  <!-- Back Button & Top Navigation -->
  <div class="flex items-center space-x-2">
    <Button variant="ghost" size="sm" onclick={() => router.navigate("models")}>
      ← Back to Models
    </Button>
  </div>

  {#if isLoading}
    <div class="p-12 text-center text-text-muted space-y-2">
      <div
        class="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto"
      ></div>
      <p class="text-xs">Loading hardware options for {modelId}...</p>
    </div>
  {:else if error}
    {#if error.toLowerCase().includes("not installed")}
      <Card variant="surface" padding="lg" class="max-w-md mx-auto text-center space-y-4 my-8">
        <div
          class="w-12 h-12 rounded-xl bg-primary-subtle text-primary flex items-center justify-center mx-auto"
        >
          <svg
            class="w-6 h-6"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            aria-hidden="true"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
            />
          </svg>
        </div>
        <div>
          <h2 class="text-base font-bold text-text">Model Not Downloaded</h2>
          <p class="text-xs text-text-muted mt-1 leading-relaxed">
            Model <strong class="text-text font-mono">{modelId}</strong> must be downloaded before you
            can configure GPU layers, K/V precision, and run hardware benchmarks.
          </p>
        </div>
        <div class="flex justify-center gap-3 pt-2">
          <Button variant="primary" size="sm" onclick={() => router.navigate("models")}>
            Go to Models to Download
          </Button>
        </div>
      </Card>
    {:else}
      <div
        role="alert"
        class="p-6 rounded-xl bg-danger-subtle border border-danger text-sm text-danger"
      >
        <p class="font-bold">Failed to load model options</p>
        <p class="text-xs mt-1">{error}</p>
        <Button variant="secondary" size="sm" class="mt-4" onclick={loadOptions}>Try again</Button>
      </div>
    {/if}
  {:else if optionsData}
    <!-- 1. Header with GPU Hardware Specs -->
    <header class="space-y-4 border-b border-border pb-6">
      <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div class="flex items-center space-x-2">
            <h1 class="text-2xl font-bold tracking-tight text-text">{optionsData.name} Options</h1>
            <Badge variant="purple">{optionsData.family}</Badge>
          </div>
          <p class="text-xs text-text-muted mt-1">
            Configure precision and speculative drafting tailored for your graphics hardware.
          </p>
        </div>

        <!-- GPU Hardware Card -->
        <div
          class="flex items-center space-x-3 bg-surface-sunken p-3 rounded-xl border border-border text-xs font-mono"
        >
          <div
            class="w-8 h-8 rounded-lg bg-primary-subtle text-primary flex items-center justify-center shrink-0"
          >
            <svg
              class="w-4 h-4"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="1.5"
              aria-hidden="true"
            >
              <rect x="4" y="4" width="16" height="16" rx="2" />
              <rect x="9" y="9" width="6" height="6" />
              <path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3" />
            </svg>
          </div>
          <div>
            <div class="font-semibold text-text">{optionsData.machine.gpu}</div>
            <div class="text-text-subtle text-[11px]">
              VRAM: {(optionsData.machine.vram_free_assumed_mb / 1024).toFixed(1)} GB free of {(
                optionsData.machine.vram_total_mb / 1024
              ).toFixed(0)} GB total
            </div>
          </div>
        </div>
      </div>

      <!-- Battery Power Warning -->
      {#if optionsData.calibration.entries_measured_on_battery > 0}
        <div
          role="alert"
          class="p-3 rounded-lg bg-warning-subtle border border-warning text-xs text-warning flex items-center space-x-2"
        >
          <span>⚠️</span>
          <span
            >Measurements were taken on battery power: speeds may be lower than when plugged into AC
            power.</span
          >
        </div>
      {/if}
    </header>

    <!-- 2. Interactive Form Controls: K/V Precision & Flash -->
    <section
      class="grid grid-cols-1 md:grid-cols-3 gap-6"
      aria-labelledby="heading-format-controls"
    >
      <div class="md:col-span-2 space-y-4">
        <h2
          id="heading-format-controls"
          class="text-sm font-bold text-text uppercase tracking-wider"
        >
          K/V Cache Precision
        </h2>

        <div
          class="grid grid-cols-1 sm:grid-cols-2 gap-3"
          role="radiogroup"
          aria-label="K/V precision options"
        >
          {#each optionsData.formats as fmt (fmt.id)}
            {@const isSelected = selectedKv === fmt.id}
            {@const qualityBadgeVariant =
              fmt.quality.label === "high"
                ? "success"
                : fmt.quality.label === "medium"
                  ? "purple"
                  : fmt.quality.label === "experimental"
                    ? "warning"
                    : "neutral"}

            <button
              type="button"
              role="radio"
              aria-checked={isSelected}
              onclick={() => (selectedKv = fmt.id)}
              class="p-4 rounded-xl border text-left transition-colors cursor-pointer select-none space-y-2 {isSelected
                ? 'bg-primary-subtle border-primary ring-1 ring-primary'
                : 'bg-surface border-border hover:bg-surface-raised'}"
            >
              <div class="flex items-center justify-between">
                <span class="font-bold text-sm text-text">{fmt.name}</span>
                <Badge variant={qualityBadgeVariant} size="sm">{fmt.quality.text}</Badge>
              </div>

              <p class="text-xs text-text-muted leading-relaxed">{fmt.description}</p>

              {#if fmt.caveat}
                <p class="text-[11px] text-warning italic leading-snug">{fmt.caveat}</p>
              {/if}
            </button>
          {/each}
        </div>
      </div>

      <!-- Flash Toggle Card -->
      <div class="space-y-4">
        <h2 class="text-sm font-bold text-text uppercase tracking-wider">Flash Acceleration</h2>
        <Card variant="surface" padding="md" class="space-y-4">
          <Toggle
            label="Enable Flash Drafter"
            description="Uses ~150 MB memory for speculative decoding, yielding significantly higher tokens per second."
            disabled={!optionsData.flash ||
              typeof optionsData.flash !== "object" ||
              !("available" in optionsData.flash) ||
              !optionsData.flash.available}
            bind:checked={selectedFlash}
          />

          {#if optionsData.flash && typeof optionsData.flash === "object" && "note" in optionsData.flash && optionsData.flash.note}
            <p class="text-xs text-text-subtle">{String(optionsData.flash.note)}</p>
          {/if}
        </Card>
      </div>
    </section>

    <!-- 3. Result Card: Active Profile Summary -->
    {#if currentOption}
      <section class="space-y-4" aria-labelledby="heading-result">
        <h2 id="heading-result" class="text-sm font-bold text-text uppercase tracking-wider">
          Hardware Placement & Speed
        </h2>

        <Card
          variant="raised"
          padding="lg"
          class="space-y-6 border-2 {isRecommended ? 'border-primary' : 'border-border'}"
        >
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div class="space-y-1">
              <div class="flex items-center space-x-2">
                <h3 class="text-lg font-bold text-text">
                  Fastest Setup: {formatContextLabel(currentOption.fast.context)} tokens of context
                </h3>
                {#if isRecommended}
                  <Badge variant="success">★ Recommended</Badge>
                {/if}
              </div>

              {#if isRecommended && optionsData.recommended.reason}
                <p class="text-xs text-success font-medium">
                  {optionsData.recommended.reason}
                </p>
              {/if}
            </div>

            <!-- Speed Box -->
            <div class="text-right">
              {#if currentOption.fast.speed.source === "measured"}
                <div class="text-2xl font-bold font-mono text-primary tabular-nums">
                  {currentOption.fast.speed.decode_tps_start.toFixed(1)}
                  <span class="text-xs font-sans text-text-muted">tok/s</span>
                </div>
                {#if currentOption.fast.speed.decode_tps_full_estimate}
                  <div class="text-[11px] text-text-subtle font-mono">
                    ~{currentOption.fast.speed.decode_tps_full_estimate.toFixed(1)} tok/s when full
                  </div>
                {/if}
              {:else}
                <div class="text-sm font-mono text-text-muted italic">Not measured yet</div>
                <Button
                  variant="ghost"
                  size="sm"
                  class="text-primary hover:text-primary text-xs"
                  onclick={() => (isBenchmarkModalOpen = true)}
                >
                  Measure on this PC
                </Button>
              {/if}
            </div>
          </div>

          <!-- 4. Context Ladder Table -->
          <div class="space-y-2">
            <span class="text-xs font-semibold text-text uppercase tracking-wider"
              >Context Ladder</span
            >
            <div class="overflow-x-auto border border-border rounded-lg bg-surface-sunken">
              <table class="w-full text-xs text-left">
                <thead
                  class="border-b border-border text-text-subtle uppercase text-[10px] font-mono"
                >
                  <tr>
                    <th class="p-3">Context</th>
                    <th class="p-3">GPU Placement</th>
                    <th class="p-3">Cache VRAM</th>
                    <th class="p-3 text-right">Generation Speed</th>
                  </tr>
                </thead>
                <tbody class="divide-y divide-border">
                  {#each currentOption.points as pt (pt.context)}
                    <tr class="hover:bg-surface-raised transition-colors">
                      <td class="p-3 font-bold font-mono text-text">
                        {formatContextLabel(pt.context)} tokens
                      </td>
                      <td class="p-3">
                        {#if pt.fully_on_gpu}
                          <Badge variant="success" size="sm">100% on GPU</Badge>
                        {:else}
                          <span class="text-warning font-mono"
                            >{pt.layers_gpu} / {pt.layers} layers on GPU</span
                          >
                        {/if}
                      </td>
                      <td class="p-3 font-mono text-text-muted">
                        {pt.kv_cache_mb} MB
                      </td>
                      <td class="p-3 text-right font-mono">
                        {#if pt.speed.source === "measured"}
                          <span class="font-bold text-text tabular-nums"
                            >{pt.speed.decode_tps_start.toFixed(1)} tok/s</span
                          >
                        {:else}
                          <span class="text-text-subtle italic">Unmeasured</span>
                        {/if}
                      </td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>
          </div>

          <!-- Bottom Actions -->
          <div class="flex items-center justify-between pt-4 border-t border-border">
            <Button variant="secondary" size="md" onclick={() => (isBenchmarkModalOpen = true)}>
              Measure Benchmarks on this PC
            </Button>

            <Button variant="primary" size="md" class="font-bold px-6" onclick={applyAndLoad}>
              Load with these settings
            </Button>
          </div>
        </Card>
      </section>
    {/if}
  {/if}

  <!-- Benchmark Dialog Modal -->
  <Dialog
    isOpen={isBenchmarkModalOpen}
    title="Run Hardware Inference Benchmark"
    description="Measures real generation and prefill speeds for this model on your graphics card."
    onclose={() => {
      if (!isBenchmarking) isBenchmarkModalOpen = false;
    }}
  >
    <div class="space-y-4 text-xs text-text-muted leading-relaxed">
      <p>
        Benchmarking takes about 30–60 seconds and will fully exercise the GPU. If you are on a
        laptop, please ensure your power adapter is plugged in for peak hardware clocks.
      </p>

      {#if activeBenchmark}
        <div
          class="space-y-2 p-3 rounded-lg bg-surface-sunken border border-border font-mono text-[11px]"
        >
          <div class="flex justify-between font-bold text-text">
            <span>Status: {activeBenchmark.status.toUpperCase()}</span>
          </div>
          <div class="space-y-1 text-text-subtle max-h-32 overflow-y-auto">
            {#each activeBenchmark.log as line, i (i)}
              <div>{line}</div>
            {/each}
          </div>
        </div>
      {/if}

      {#if isBenchmarking}
        <ProgressBar label="Running benchmarks..." />
      {/if}
    </div>

    {#snippet actions()}
      {#if isBenchmarking}
        <Button variant="danger" size="sm" onclick={cancelBenchmark}>Cancel Benchmark</Button>
      {:else}
        <Button variant="ghost" size="sm" onclick={() => (isBenchmarkModalOpen = false)}
          >Close</Button
        >
        <Button variant="primary" size="sm" onclick={startBenchmark}>Start Benchmark</Button>
      {/if}
    {/snippet}
  </Dialog>
</div>
