import React, { useCallback, useEffect, useState } from "react";
import { useModelsStore } from "../lib/stores/useModelsStore.ts";
import { useRouter } from "../lib/router/useRouter.ts";
import { Icon } from "../lib/ui/Icon.tsx";
import { Dialog } from "../lib/ui/Dialog.tsx";
import { Button } from "../lib/ui/Button.tsx";
import { Skeleton } from "../lib/ui/Skeleton.tsx";
import { EmptyState } from "../lib/ui/EmptyState.tsx";
import { ErrorState } from "../lib/ui/ErrorState.tsx";
import { toast } from "../lib/ui/Toast.tsx";
import { ModelCard } from "../lib/features/models/ModelCard.tsx";
import { installedFiles, installedSizeMb } from "../lib/features/models/modelHelpers.ts";
import { formatGb, formatMb } from "../lib/utils/format.ts";

function messageOf(err: unknown, fallback: string): string {
  return err instanceof Error && err.message ? err.message : fallback;
}

export const ModelsPage: React.FC = () => {
  const [modelToDelete, setModelToDelete] = useState<string | null>(null);

  const models = useModelsStore((s) => s.models);
  const isLoading = useModelsStore((s) => s.isLoading);
  const downloads = useModelsStore((s) => s.downloads);
  const diskFreeGb = useModelsStore((s) => s.diskFreeGb);
  const error = useModelsStore((s) => s.error);
  const notice = useModelsStore((s) => s.notice);
  const loadingModelId = useModelsStore((s) => s.loadingModelId);
  const loadProgress = useModelsStore((s) => s.loadProgress);
  const fetchModels = useModelsStore((s) => s.fetchModels);
  const downloadModel = useModelsStore((s) => s.downloadModel);
  const downloadAddon = useModelsStore((s) => s.downloadAddon);
  const cancelDownload = useModelsStore((s) => s.cancelDownload);
  const loadModel = useModelsStore((s) => s.loadModel);
  const unloadModel = useModelsStore((s) => s.unloadModel);
  const deleteModel = useModelsStore((s) => s.deleteModel);
  const clearNotice = useModelsStore((s) => s.clearNotice);
  const navigate = useRouter((s) => s.navigate);

  useEffect(() => {
    void fetchModels();
  }, [fetchModels]);

  const closeDelete = useCallback(() => setModelToDelete(null), []);
  const target = models.find((m) => m.id === modelToDelete);

  const run = async (action: () => Promise<void>, fallback: string): Promise<void> => {
    try {
      await action();
    } catch (err: unknown) {
      toast.show(messageOf(err, fallback), "error");
    }
  };

  const executeDelete = async (): Promise<void> => {
    if (!target) return;
    const { id, dara } = target;
    const freed = formatMb(installedSizeMb(target));
    setModelToDelete(null);
    await run(async () => {
      await deleteModel(id);
      toast.show(`${dara.name} was deleted from disk. ${freed} freed.`, "info");
    }, "Could not delete the model");
  };

  return (
    <div className="p-6 max-w-6xl w-full mx-auto space-y-6 text-left select-text">
      <header className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-border pb-5">
        <div>
          <h1 className="text-xl font-bold text-text">Models</h1>
          <p className="text-sm text-text-muted mt-0.5">
            Download a model once, then it runs on your own graphics card.
          </p>
        </div>
        {diskFreeGb !== null && (
          <div className="flex items-center gap-2 text-xs text-text-muted bg-surface-raised px-3 py-1.5 rounded-lg border border-border">
            <Icon name="hard-drive" size={13} className="text-primary" />
            <span>
              Disk free: <strong className="text-text font-bold">{formatGb(diskFreeGb)}</strong>
            </span>
          </div>
        )}
      </header>

      {notice && (
        <div
          role="status"
          className="p-3 rounded-xl bg-warning-subtle border border-border text-xs text-text flex items-start gap-2"
        >
          <Icon name="info" size={14} className="text-warning shrink-0 mt-0.5" />
          <span className="flex-1">{notice}</span>
          <button
            type="button"
            onClick={clearNotice}
            aria-label="Dismiss notice"
            className="text-text-muted hover:text-text cursor-pointer"
          >
            <Icon name="x" size={14} />
          </button>
        </div>
      )}

      {error && models.length > 0 && (
        <div
          role="alert"
          className="p-3 rounded-xl bg-danger-subtle border border-border text-xs text-danger font-medium flex items-center gap-2"
        >
          <Icon name="alert-triangle" size={16} />
          <span>{error}</span>
        </div>
      )}

      {models.length === 0 && isLoading && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5" aria-busy="true">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-96 w-full" />
          ))}
        </div>
      )}

      {models.length === 0 && !isLoading && error && (
        <ErrorState
          title="Could not load the models"
          message={error}
          onRetry={() => void fetchModels()}
        />
      )}

      {models.length === 0 && !isLoading && !error && (
        <EmptyState
          icon="box"
          title="No models listed"
          description="The engine did not report any models."
        />
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
        {models.map((model) => (
          <ModelCard
            key={model.id}
            model={model}
            job={downloads[model.id]}
            diskFreeGb={diskFreeGb}
            isLoadingThis={loadingModelId === model.id}
            loadProgress={loadProgress}
            isBusy={loadingModelId !== null}
            onDownload={() => void run(() => downloadModel(model.id), "Download failed")}
            onAddon={(kind) => void run(() => downloadAddon(model.id, kind), "Download failed")}
            onCancel={() => void cancelDownload(model.id)}
            onLoad={() => void run(() => loadModel(model.id), "Could not load the model")}
            onUnload={() => void unloadModel()}
            onDelete={() => setModelToDelete(model.id)}
            onOptions={() => navigate("options", model.id)}
            onChat={() => navigate("chat")}
          />
        ))}
      </div>

      <Dialog
        isOpen={target !== undefined}
        title={target ? `Delete ${target.dara.name}?` : "Delete model"}
        description="These files will be removed from your disk. You can download them again later."
        onclose={closeDelete}
        actions={
          <>
            <Button variant="ghost" size="sm" onClick={closeDelete}>
              Cancel
            </Button>
            <Button variant="danger" size="sm" onClick={() => void executeDelete()}>
              Delete files
            </Button>
          </>
        }
      >
        {target && (
          <>
            <ul className="list-disc pl-5 text-xs">
              {installedFiles(target).map((f) => (
                <li key={f.kind}>
                  {f.name} ({formatMb(f.size_mb)})
                </li>
              ))}
            </ul>
            <p className="text-xs font-semibold text-text">
              Disk space freed: {formatMb(installedSizeMb(target))}
            </p>
          </>
        )}
      </Dialog>
    </div>
  );
};
