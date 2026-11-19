<script lang="ts">
  import { onMount } from "svelte";
  import { modelsStore } from "../lib/stores/models.svelte.ts";
  import { router } from "../lib/router/router.svelte.ts";
  import Icon from "../lib/ui/Icon.svelte";
  import Dialog from "../lib/ui/Dialog.svelte";
  import Button from "../lib/ui/Button.svelte";
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
      toast.show(`Model ${id} was deleted from disk.`, "info");
    }
  }

  function handleCancelDownload(modelId: string) {
    void modelsStore.cancelDownload(modelId);
    toast.show("Download paused. Partial chunks saved on disk.", "info");
  }

  function getModelFileSize(model: (typeof modelsStore.models)[0]): string {
    const modelFile = model.dara.files?.find((f) => f.kind === "model");
    if (modelFile && modelFile.size_mb > 0) {
      return (modelFile.size_mb / 1024).toFixed(2) + " GB";
    }
    return (model.dara.download_size_mb / 1024).toFixed(2) + " GB";
  }

  function getTotalDownloadSize(model: (typeof modelsStore.models)[0]): string {
    return (model.dara.download_size_mb / 1024).toFixed(2) + " GB";
  }
</script>

<div class="p-6 max-w-6xl w-full mx-auto space-y-6 text-left">
  <!-- Header Bar -->
  <header class="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-border pb-5">
    <div>
      <h1 class="text-sm font-bold text-text-subtle uppercase tracking-widest">Model Management</h1>
      <p class="text-xs text-text-muted mt-0.5">
        Local weight files, GPU placement planners, and multimodal vision towers
      </p>
    </div>

    {#if modelsStore.diskFreeGb !== null}
      <div class="flex items-center gap-2 text-xs font-mono text-text-muted bg-surface-raised px-3 py-1.5 rounded-lg border border-border">
        <Icon name="hard-drive" size={13} class="text-primary" />
        <span>Disk Free: <strong class="text-text font-bold">{modelsStore.diskFreeGb.toFixed(1)} GB</strong></span>
      </div>
    {/if}
  </header>

  {#if modelsStore.error}
    <div role="alert" class="p-4 rounded-xl bg-danger/10 border border-danger/30 text-xs text-danger font-medium flex items-center gap-2">
      <Icon name="alert-circle" size={16} />
      <span>{modelsStore.error}</span>
    </div>
  {/if}

  <!-- Models Grid (Oprel ModelsView Style) -->
  <div class="grid grid-cols-1 md:grid-cols-3 gap-5">
    {#each modelsStore.models as model (model.id)}
      {@const job = modelsStore.downloads[model.id]}
      {@const isDownloading = Boolean(
        job &&
        (job.status === "running" ||
          job.status === "verifying" ||
          job.status === "queued" ||
          job.status === "downloading")
      )}
      {@const isLoaded = model.dara.loaded}
      {@const isInstalled = model.dara.status === "installed"}
      {@const fitsGpu = model.dara.fits_gpu}

      <div
        class="flex flex-col justify-between bg-surface border rounded-xl p-5 transition-all shadow-sm {isLoaded
          ? 'border-primary/60 ring-1 ring-primary/40 bg-surface-raised'
          : 'border-border hover:border-primary/30'}"
      >
        <!-- Top Half -->
        <div class="space-y-4">
          <!-- Card Header: Title & Badges -->
          <div class="flex items-start justify-between gap-2">
            <div>
              <h2 class="text-base font-bold text-text tracking-tight">{model.dara.name}</h2>
              <p class="text-[11px] text-text-subtle mt-0.5">{model.dara.base_model}</p>
            </div>

            <div class="flex flex-col items-end gap-1 shrink-0">
              <!-- Status Badge (Oprel Style) -->
              {#if isLoaded}
                <span class="flex items-center gap-1.5 text-[10px] font-bold text-success px-2 py-0.5 rounded-md bg-success/10 border border-success/20">
                  <span class="w-1.5 h-1.5 rounded-full bg-success animate-pulse"></span>
                  LOADED
                </span>
              {:else if isDownloading}
                <span class="text-[10px] font-bold text-warning animate-pulse px-2 py-0.5 rounded-md bg-warning/10 border border-warning/20">
                  DOWNLOADING
                </span>
              {:else if isInstalled}
                <span class="text-[10px] font-bold text-primary px-2 py-0.5 rounded-md bg-primary/10 border border-primary/20">
                  READY
                </span>
              {:else}
                <span class="text-[10px] font-bold text-text-subtle uppercase px-2 py-0.5 rounded-md bg-surface-sunken border border-border/60">
                  REGISTRY
                </span>
              {/if}

              <!-- GPU Fit Badge -->
              {#if fitsGpu === "full"}
                <span class="text-[10px] font-bold text-success">Compatible (Full GPU)</span>
              {:else if fitsGpu === "partial"}
                <span class="text-[10px] font-bold text-warning">Partial (CPU Offload)</span>
              {:else}
                <span class="text-[10px] font-bold text-danger">Exceeds VRAM</span>
              {/if}
            </div>
          </div>

          <!-- Description -->
          <p class="text-xs text-text-muted leading-relaxed line-clamp-2">
            {model.dara.description}
          </p>

          <!-- Capability Badges -->
          <div class="flex flex-wrap gap-1.5 pt-1">
            <span class="px-2 py-0.5 rounded-md bg-surface-raised border border-border text-[10px] font-mono text-text-subtle">
              {model.dara.quant}
            </span>
            {#if model.dara.flash?.available}
              <span class="px-2 py-0.5 rounded-md bg-primary/10 border border-primary/20 text-[10px] font-semibold text-primary flex items-center gap-1">
                <Icon name="zap" size={10} />
                Flash Drafter
              </span>
            {/if}
            {#if model.dara.vision?.available}
              <span class="px-2 py-0.5 rounded-md bg-success/10 border border-success/20 text-[10px] font-semibold text-success flex items-center gap-1">
                <Icon name="image" size={10} />
                Vision Tower
              </span>
            {/if}
          </div>

          <!-- Specs Table (Endpoint Real Data) -->
          <div class="space-y-1.5 pt-3 border-t border-border/50 text-xs">
            <div class="flex justify-between py-0.5">
              <span class="text-text-subtle">Model Weights:</span>
              <span class="font-mono font-medium text-text">{getModelFileSize(model)}</span>
            </div>
            <div class="flex justify-between py-0.5">
              <span class="text-text-subtle">Full Download (all files):</span>
              <span class="font-mono font-medium text-text">{getTotalDownloadSize(model)}</span>
            </div>
            <div class="flex justify-between py-0.5">
              <span class="text-text-subtle">VRAM Requirement:</span>
              <span class="font-mono text-text">
                {model.dara.min_vram_mb} MB - {model.dara.full_vram_mb} MB
              </span>
            </div>
          </div>
        </div>

        <!-- Bottom Half: Download Progress & Actions -->
        <div class="pt-5 space-y-3">
          {#if isDownloading && job}
            <!-- Live Download Progress Panel -->
            <div class="p-3 bg-surface-sunken border border-border rounded-xl space-y-2">
              <div class="flex justify-between items-center text-xs">
                <span class="font-bold text-text truncate max-w-[160px]">
                  {job.file || "Downloading..."}
                </span>
                <span class="font-mono font-bold text-primary tabular-nums">
                  {job.percent.toFixed(1)}%
                </span>
              </div>

              <!-- Segmented Progress Bar -->
              <div class="h-2 w-full bg-surface-raised rounded-full overflow-hidden border border-border/60">
                <div
                  class="h-full bg-primary transition-all duration-300 rounded-full"
                  style="width: {Math.max(2, job.percent)}%"
                ></div>
              </div>

              <div class="flex justify-between items-center text-[10px] font-mono text-text-subtle">
                <span>{(job.completed_bytes / 1048576).toFixed(0)} / {(job.total_bytes / 1048576).toFixed(0)} MB</span>
                <span>{job.mb_per_second.toFixed(1)} MB/s</span>
              </div>

              <button
                type="button"
                onclick={() => handleCancelDownload(model.id)}
                class="w-full py-1 text-center text-[11px] font-semibold text-danger hover:bg-danger/10 rounded-md transition-colors cursor-pointer"
              >
                Cancel Download
              </button>
            </div>
          {:else if isLoaded}
            <!-- Loaded Model Actions -->
            <div class="flex items-center gap-2">
              <button
                type="button"
                onclick={() => modelsStore.unloadModel()}
                class="flex-1 h-9 px-3 rounded-lg border border-border bg-surface hover:bg-surface-raised text-xs font-semibold text-text transition-colors cursor-pointer"
              >
                Unload
              </button>
              <button
                type="button"
                onclick={() => confirmDelete(model.id)}
                class="h-9 px-3 rounded-lg border border-danger/30 text-danger hover:bg-danger/10 transition-colors cursor-pointer"
                title="Delete model from disk"
              >
                <Icon name="trash-2" size={14} />
              </button>
            </div>
          {:else if isInstalled}
            <!-- Installed Model Actions -->
            <div class="flex items-center gap-2">
              <button
                type="button"
                onclick={() => modelsStore.loadModel(model.id)}
                class="flex-1 h-9 px-4 rounded-lg bg-primary hover:bg-primary-hover text-white text-xs font-semibold transition-all cursor-pointer shadow-sm"
              >
                Load into VRAM
              </button>
              <button
                type="button"
                onclick={() => confirmDelete(model.id)}
                class="h-9 px-3 rounded-lg border border-border hover:border-danger/40 hover:text-danger text-text-subtle transition-colors cursor-pointer"
                title="Delete model from disk"
              >
                <Icon name="trash-2" size={14} />
              </button>
            </div>
          {:else}
            <!-- Uninstalled Model Actions -->
            <button
              type="button"
              onclick={() => modelsStore.downloadModel(model.id)}
              class="w-full h-9 px-4 rounded-lg bg-primary hover:bg-primary-hover text-white text-xs font-semibold transition-all cursor-pointer flex items-center justify-center gap-2 shadow-sm"
            >
              <Icon name="download" size={14} />
              <span>Download & Install ({getTotalDownloadSize(model)})</span>
            </button>
          {/if}

          <!-- Options link -->
          <button
            type="button"
            onclick={() => router.navigate("options", model.id)}
            class="w-full py-1 text-center text-[11px] font-medium text-text-muted hover:text-text hover:underline transition-colors cursor-pointer"
          >
            Configure Model Options →
          </button>
        </div>
      </div>
    {/each}
  </div>

  <!-- Delete Confirmation Dialog -->
  <Dialog
    isOpen={modelToDelete !== null}
    title="Delete Model Weights"
    onclose={() => (modelToDelete = null)}
  >
    <div class="space-y-3 text-xs text-text-muted">
      <p>
        Are you sure you want to delete <strong class="text-text font-bold">{modelToDelete}</strong>?
      </p>
      <p>
        This will remove the downloaded GGUF weights, Flash drafters, and vision towers from disk. You can download them again anytime.
      </p>
    </div>

    {#snippet actions()}
      <Button variant="ghost" size="sm" onclick={() => (modelToDelete = null)}>Cancel</Button>
      <Button variant="danger" size="sm" onclick={executeDelete}>Delete Model</Button>
    {/snippet}
  </Dialog>
</div>
