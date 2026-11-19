<script lang="ts">
  import { onMount } from "svelte";
  import { modelsStore } from "../lib/stores/models.svelte.ts";
  import { router } from "../lib/router/router.svelte.ts";
  import Icon from "../lib/ui/Icon.svelte";

  onMount(() => {
    void modelsStore.fetchModels();
  });

  async function handleDownloadAndLoad(modelId: string) {
    await modelsStore.downloadModel(modelId, ["model"]);
    const checkInterval = setInterval(async () => {
      const job = modelsStore.downloads[modelId];
      if (job && job.status === "done") {
        clearInterval(checkInterval);
        await modelsStore.loadModel(modelId);
        const loadCheck = setInterval(() => {
          if (modelsStore.loadProgress?.state === "ready" || modelsStore.loadedModel) {
            clearInterval(loadCheck);
            router.navigate("chat");
          }
        }, 300);
      }
    }, 500);
  }
</script>

<div class="flex-1 flex flex-col items-center justify-center p-8 max-w-5xl mx-auto space-y-8 text-left">
  <header class="text-center space-y-3 max-w-xl">
    <div class="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-primary/10 border border-primary/25 text-primary mx-auto mb-2 shadow-sm">
      <Icon name="bot" size={28} />
    </div>
    <h1 class="text-3xl font-bold tracking-tight text-text">Welcome to Dara Studio</h1>
    <p class="text-sm text-text-muted leading-relaxed">
      Dara runs intelligent language models directly on your Windows GPU. Choose a model below to
      download and begin chatting privately offline.
    </p>
  </header>

  {#if modelsStore.error}
    <div role="alert" class="w-full max-w-md p-4 rounded-xl bg-danger/10 border border-danger/30 text-xs text-danger font-medium flex items-center gap-2">
      <Icon name="alert-circle" size={16} />
      <span>{modelsStore.error}</span>
    </div>
  {/if}

  <!-- 3 Model Tiers Grid -->
  <div class="grid grid-cols-1 md:grid-cols-3 gap-5 w-full">
    {#each modelsStore.models as model (model.id)}
      {@const job = modelsStore.downloads[model.id]}
      {@const isDownloading = Boolean(
        job &&
        (job.status === "running" ||
          job.status === "verifying" ||
          job.status === "queued" ||
          job.status === "downloading")
      )}
      {@const isRecommended = model.dara.fits_gpu === "full" || model.id === "dara-nano"}

      <div
        class="flex flex-col justify-between bg-surface border rounded-xl p-5 transition-all shadow-sm {isRecommended
          ? 'border-primary/60 ring-1 ring-primary/40 bg-surface-raised'
          : 'border-border hover:border-primary/30'}"
      >
        <div class="space-y-4">
          <div class="flex items-center justify-between">
            <h2 class="text-base font-bold text-text">{model.dara.name}</h2>
            {#if isRecommended}
              <span class="px-2 py-0.5 rounded-md text-[10px] font-bold bg-success/10 text-success border border-success/20">
                Recommended
              </span>
            {/if}
          </div>

          <p class="text-xs text-text-muted leading-relaxed min-h-[44px]">
            {model.dara.description}
          </p>

          <div class="p-3 rounded-xl bg-surface-sunken border border-border space-y-1.5 text-xs font-mono">
            <div class="flex justify-between">
              <span class="text-text-subtle">Download:</span>
              <span class="font-medium text-text">{(model.dara.download_size_mb / 1024).toFixed(1)} GB</span>
            </div>
            <div class="flex justify-between">
              <span class="text-text-subtle">GPU Fit:</span>
              <span class="font-medium capitalize text-text">{model.dara.fits_gpu}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-text-subtle">Quant:</span>
              <span class="font-medium text-text">{model.dara.quant}</span>
            </div>
          </div>
        </div>

        <div class="space-y-3 pt-4">
          {#if isDownloading && job}
            <div class="space-y-2">
              <div class="flex justify-between text-xs font-mono">
                <span class="font-bold text-text truncate max-w-[140px]">{job.file || "Downloading..."}</span>
                <span class="font-bold text-primary">{job.percent.toFixed(1)}%</span>
              </div>
              <div class="h-2 w-full bg-surface-sunken rounded-full overflow-hidden border border-border">
                <div class="h-full bg-primary transition-all duration-300 rounded-full" style="width: {Math.max(2, job.percent)}%"></div>
              </div>
              <div class="flex justify-between text-[10px] font-mono text-text-subtle">
                <span>{job.mb_per_second.toFixed(1)} MB/s</span>
                <span>{job.eta_seconds > 0 ? Math.round(job.eta_seconds) + "s ETA" : ""}</span>
              </div>
              <button
                type="button"
                onclick={() => modelsStore.cancelDownload(model.id)}
                class="w-full py-1 text-center text-[11px] font-semibold text-danger hover:underline cursor-pointer"
              >
                Cancel
              </button>
            </div>
          {:else}
            <button
              type="button"
              onclick={() => handleDownloadAndLoad(model.id)}
              class="w-full h-9 px-4 rounded-lg bg-primary hover:bg-primary-hover text-white text-xs font-semibold transition-all cursor-pointer shadow-sm flex items-center justify-center gap-2"
            >
              <Icon name="download" size={14} />
              <span>Download & Start</span>
            </button>
          {/if}
        </div>
      </div>
    {/each}
  </div>
</div>
