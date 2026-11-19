<script lang="ts">
  import { onMount } from "svelte";
  import { modelsStore } from "../lib/stores/models.svelte.ts";
  import { router } from "../lib/router/router.svelte.ts";
  import Card from "../lib/ui/Card.svelte";
  import Badge from "../lib/ui/Badge.svelte";
  import Button from "../lib/ui/Button.svelte";
  import ProgressBar from "../lib/ui/ProgressBar.svelte";
  import Dialog from "../lib/ui/Dialog.svelte";
  import { toast } from "../lib/ui/Toast.svelte";

  let modelToDelete = $state<string | null>(null);

  onMount(() => {
    void modelsStore.fetchModels();
  });

  function confirmDelete(modelId: string) {
    modelToDelete = modelId;
  }

  async function executeDelete() {
    if (modelToDelete) {
      const id = modelToDelete;
      modelToDelete = null;
      await modelsStore.deleteModel(id);
      toast.show(`Model ${id} deleted`, "info");
    }
  }

  function handleCancelDownload(modelId: string) {
    void modelsStore.cancelDownload(modelId);
    toast.show("Download cancelled. Partial file saved for resuming.", "info");
  }
</script>

<div class="p-8 max-w-5xl mx-auto space-y-8 text-left">
  <!-- Header with Disk Status -->
  <header
    class="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-border pb-6"
  >
    <div class="space-y-1">
      <h1 class="text-2xl font-bold tracking-tight text-text">Models</h1>
      <p class="text-sm text-text-muted">
        Manage local model weights, quantizations, and GPU allocations.
      </p>
    </div>

    {#if modelsStore.diskFreeGb !== null}
      <div
        class="flex items-center space-x-2 text-xs font-mono text-text-muted bg-surface-sunken p-2.5 rounded-lg border border-border"
      >
        <svg
          class="w-4 h-4 text-primary shrink-0"
          viewBox="0 0 20 20"
          fill="none"
          stroke="currentColor"
          stroke-width="1.5"
          aria-hidden="true"
        >
          <path d="M4 6a2 2 0 012-2h8a2 2 0 012 2v8a2 2 0 01-2 2H6a2 2 0 01-2-2V6z" />
          <circle cx="8" cy="10" r="1" fill="currentColor" />
        </svg>
        <span>Disk Free: <strong>{modelsStore.diskFreeGb.toFixed(1)} GB</strong></span>
      </div>
    {/if}
  </header>

  {#if modelsStore.error}
    <div
      role="alert"
      class="p-4 rounded-xl bg-danger-subtle border border-danger text-xs text-danger font-medium"
    >
      {modelsStore.error}
    </div>
  {/if}

  <!-- Models Grid -->
  <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
    {#each modelsStore.models as model (model.id)}
      {@const job = modelsStore.downloads[model.id]}
      {@const isDownloading = Boolean(
        job &&
        (job.status === "running" ||
          job.status === "verifying" ||
          job.status === "queued" ||
          job.status === "downloading")
      )}
      {@const isLoading = modelsStore.loadingModelId === model.id}
      {@const isLoaded = model.dara.loaded}
      {@const isInstalled = model.dara.status === "installed"}
      {@const isRecommended = model.dara.fits_gpu === "full"}

      <Card
        variant={isLoaded ? "raised" : "surface"}
        padding="lg"
        class="flex flex-col justify-between space-y-6 relative {isLoaded
          ? 'border-primary border-2 shadow-sm'
          : ''}"
      >
        <div class="space-y-4">
          <!-- Card Header -->
          <div class="flex items-start justify-between gap-2">
            <div>
              <h2 class="text-lg font-bold text-text">{model.dara.name}</h2>
              <span class="text-xs text-text-subtle font-mono">{model.dara.quant}</span>
            </div>

            <div class="flex flex-col items-end space-y-1">
              {#if isLoaded}
                <Badge variant="purple">Loaded</Badge>
              {:else if isInstalled}
                <Badge variant="success">Installed</Badge>
              {:else if isDownloading}
                <Badge variant="warning">Downloading</Badge>
              {:else}
                <Badge variant="neutral">Not installed</Badge>
              {/if}

              {#if isRecommended && !isLoaded}
                <span class="text-[10px] font-semibold text-success uppercase">Recommended</span>
              {/if}
            </div>
          </div>

          <!-- Description -->
          <p class="text-xs text-text-muted leading-relaxed min-h-[48px]">
            {model.dara.description}
          </p>

          <!-- Feature Chips: Flash & Vision -->
          <div class="flex flex-wrap gap-1.5 pt-1">
            {#if model.dara.flash.available}
              <Badge variant="purple" size="sm">Flash Drafter</Badge>
            {:else}
              <span class="text-[10px] text-text-subtle italic"
                >Flash: not available for this model</span
              >
            {/if}

            {#if model.dara.vision.available}
              <Badge variant="neutral" size="sm">Vision / OCR</Badge>
            {/if}
          </div>

          <!-- Specs Table -->
          <div
            class="p-3 rounded-lg bg-surface-sunken border border-border space-y-1.5 text-xs font-mono"
          >
            <div class="flex justify-between">
              <span class="text-text-subtle">Download Size:</span>
              <span class="font-medium text-text"
                >{(model.dara.download_size_mb / 1024).toFixed(1)} GB</span
              >
            </div>
            <div class="flex justify-between">
              <span class="text-text-subtle">Fits GPU:</span>
              <span class="font-medium capitalize text-text">{model.dara.fits_gpu}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-text-subtle">Context:</span>
              <span class="font-medium text-text"
                >{Math.round(model.dara.context / 1024)}K tokens</span
              >
            </div>
          </div>
        </div>

        <!-- Action Buttons -->
        <div class="space-y-3 pt-2">
          {#if isDownloading && job}
            <div class="space-y-2">
              <ProgressBar
                value={job.percent}
                max={100}
                showPercent={true}
                label="Downloading files..."
              />
              <div class="flex justify-between text-[11px] font-mono text-text-subtle">
                <span>{job.mb_per_second.toFixed(1)} MB/s</span>
                <span>ETA: {job.eta_seconds}s</span>
              </div>
              <Button
                variant="ghost"
                size="sm"
                class="w-full text-danger hover:text-danger"
                onclick={() => handleCancelDownload(model.id)}
              >
                Cancel download
              </Button>
            </div>
          {:else if isLoading}
            <div class="space-y-2">
              <ProgressBar label="Loading model into VRAM..." />
              <p class="text-xs text-center text-text-muted animate-pulse">Allocating GPU layers</p>
            </div>
          {:else if isLoaded}
            <div class="flex items-center space-x-2">
              <Button
                variant="secondary"
                size="sm"
                class="flex-1"
                onclick={() => modelsStore.unloadModel()}
              >
                Unload
              </Button>
              <Button
                variant="ghost"
                size="sm"
                class="text-danger hover:text-danger"
                onclick={() => confirmDelete(model.id)}
              >
                Delete
              </Button>
            </div>
          {:else if isInstalled}
            <div class="flex items-center space-x-2">
              <Button
                variant="primary"
                size="sm"
                class="flex-1"
                onclick={() => modelsStore.loadModel(model.id)}
              >
                Load
              </Button>
              <Button
                variant="ghost"
                size="sm"
                class="text-danger hover:text-danger"
                onclick={() => confirmDelete(model.id)}
              >
                Delete
              </Button>
            </div>
          {:else}
            <Button
              variant={isRecommended ? "primary" : "secondary"}
              size="md"
              class="w-full font-semibold"
              onclick={() => modelsStore.downloadModel(model.id)}
            >
              Download ({(model.dara.download_size_mb / 1024).toFixed(1)} GB)
            </Button>
          {/if}
          <Button
            variant="ghost"
            size="sm"
            class="w-full text-xs text-text-muted hover:text-text"
            onclick={() => router.navigate("options", model.id)}
          >
            Options & Hardware Benchmarks ⚙
          </Button>
        </div>
      </Card>
    {/each}
  </div>

  <!-- Delete Confirmation Modal Dialog -->
  <Dialog
    isOpen={modelToDelete !== null}
    title="Delete Model Files"
    description="This will permanently delete the weights from disk to free space."
    onclose={() => (modelToDelete = null)}
  >
    <p class="text-xs text-text-muted">
      Are you sure you want to delete model <strong>{modelToDelete}</strong>?
    </p>

    {#snippet actions()}
      <Button variant="ghost" size="sm" onclick={() => (modelToDelete = null)}>Cancel</Button>
      <Button variant="danger" size="sm" onclick={executeDelete}>Delete Model</Button>
    {/snippet}
  </Dialog>
</div>
