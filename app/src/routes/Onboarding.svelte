<script lang="ts">
  import { onMount } from "svelte";
  import { modelsStore } from "../lib/stores/models.svelte.ts";
  import { router } from "../lib/router/router.svelte.ts";
  import Card from "../lib/ui/Card.svelte";
  import Badge from "../lib/ui/Badge.svelte";
  import Button from "../lib/ui/Button.svelte";
  import ProgressBar from "../lib/ui/ProgressBar.svelte";

  onMount(() => {
    void modelsStore.fetchModels();
  });

  async function handleDownloadAndLoad(modelId: string) {
    await modelsStore.downloadModel(modelId, ["model"]);
    // Monitor until download finishes then auto-load
    const checkInterval = setInterval(async () => {
      const job = modelsStore.downloads[modelId];
      if (job && job.status === "done") {
        clearInterval(checkInterval);
        await modelsStore.loadModel(modelId);
        // Once loaded, redirect to chat
        const loadCheck = setInterval(() => {
          if (modelsStore.loadProgress?.state === "ready") {
            clearInterval(loadCheck);
            router.navigate("chat");
          }
        }, 300);
      }
    }, 500);
  }
</script>

<div
  class="flex-1 flex flex-col items-center justify-center p-8 max-w-4xl mx-auto space-y-8 text-left"
>
  <header class="text-center space-y-3 max-w-xl">
    <div
      class="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-primary-subtle text-primary mx-auto mb-2"
    >
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
    </div>
    <h1 class="text-3xl font-bold tracking-tight text-text">Welcome to Dara</h1>
    <p class="text-sm text-text-muted leading-relaxed">
      Dara runs intelligent language models directly on your Windows GPU. Choose a model below to
      download and begin chatting privately offline.
    </p>
  </header>

  {#if modelsStore.error}
    <div
      role="alert"
      class="w-full p-4 rounded-xl bg-danger-subtle border border-danger text-xs text-danger font-medium"
    >
      {modelsStore.error}
    </div>
  {/if}

  <!-- 3 Model Tiers Grid -->
  <div class="grid grid-cols-1 md:grid-cols-3 gap-6 w-full">
    {#each modelsStore.models as model (model.id)}
      {@const job = modelsStore.downloads[model.id]}
      {@const isDownloading = Boolean(job && job.status === "downloading")}
      {@const isLoading = modelsStore.loadingModelId === model.id}
      {@const isRecommended = model.dara.fits_gpu === "full" || model.id === "dara-nano"}

      <Card
        variant={isRecommended ? "raised" : "surface"}
        padding="lg"
        class="flex flex-col justify-between space-y-5 relative {isRecommended
          ? 'border-primary border-2 shadow-sm'
          : ''}"
      >
        <div class="space-y-4">
          <div class="flex items-center justify-between">
            <h2 class="text-lg font-bold text-text">{model.dara.name}</h2>
            {#if isRecommended}
              <Badge variant="success">Recommended</Badge>
            {/if}
          </div>

          <p class="text-xs text-text-muted leading-relaxed min-h-[48px]">
            {model.dara.description}
          </p>

          <div
            class="p-3 rounded-lg bg-surface-sunken border border-border space-y-1.5 text-xs font-mono"
          >
            <div class="flex justify-between">
              <span class="text-text-subtle">Download:</span>
              <span class="font-medium text-text"
                >{(model.dara.download_size_mb / 1024).toFixed(1)} GB</span
              >
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

        <div class="space-y-3 pt-2">
          {#if isDownloading && job}
            <div class="space-y-1.5">
              <ProgressBar
                value={job.percent}
                max={100}
                showPercent={true}
                label="Downloading..."
              />
              <div class="flex justify-between text-[11px] font-mono text-text-subtle">
                <span>{job.mb_per_second.toFixed(1)} MB/s</span>
                <span>ETA: {job.eta_seconds}s</span>
              </div>
              <Button
                variant="ghost"
                size="sm"
                class="w-full text-danger hover:text-danger"
                onclick={() => modelsStore.cancelDownload(model.id)}
              >
                Cancel
              </Button>
            </div>
          {:else if isLoading}
            <div class="space-y-1.5">
              <ProgressBar label="Loading model into VRAM..." />
              <p class="text-xs text-center text-text-muted animate-pulse">Initializing weights</p>
            </div>
          {:else}
            <Button
              variant={isRecommended ? "primary" : "secondary"}
              size="md"
              class="w-full font-semibold"
              onclick={() => handleDownloadAndLoad(model.id)}
            >
              Download & Start
            </Button>
          {/if}
        </div>
      </Card>
    {/each}
  </div>
</div>
