import React, { useState } from "react";
import type { KvFormat, ModelInfo } from "../../api/types.ts";
import { storeApi } from "../../api/store-api.ts";
import { useModelsStore } from "../../stores/useModelsStore.ts";
import { Button } from "../../ui/Button.tsx";
import { ProgressBar } from "../../ui/ProgressBar.tsx";
import { toast } from "../../ui/Toast.tsx";
import { LoadedConfigSummary } from "../models/ModelFiles.tsx";

interface Props {
  modelId: string;
  model: ModelInfo | undefined;
  kv: KvFormat;
  flash: boolean;
  vision: boolean;
  isBlocked: boolean;
}

export const ApplyPanel: React.FC<Props> = ({ modelId, model, kv, flash, vision, isBlocked }) => {
  const [failure, setFailure] = useState<string | null>(null);
  const loadModel = useModelsStore((s) => s.loadModel);
  const loadingModelId = useModelsStore((s) => s.loadingModelId);
  const loadProgress = useModelsStore((s) => s.loadProgress);
  const fetchModels = useModelsStore((s) => s.fetchModels);
  const otherLoaded = useModelsStore((s) =>
    s.models.find((m) => m.dara.loaded && m.id !== modelId)
  );

  const isLoadingThis = loadingModelId === modelId;
  const isInstalled = model?.dara.status === "installed";

  const apply = async (): Promise<void> => {
    setFailure(null);
    try {
      await loadModel(modelId, { kv, flash, vision });
    } catch (err: unknown) {
      const message =
        err instanceof Error && err.message ? err.message : "The model could not be loaded.";
      setFailure(`${message} The engine keeps the previous model loaded.`);
      return;
    }
    try {
      await storeApi.putModelConfig(modelId, { kv, flash: flash ? "on" : "off", vision });
    } catch {
      toast.show("Loaded, but your choice could not be saved for next time.", "error");
    }
    await fetchModels();
    toast.show(`${model?.dara.name ?? modelId} is loaded with your settings.`, "success");
  };

  return (
    <section
      aria-labelledby="apply-title"
      className="rounded-2xl border border-border bg-surface p-4 space-y-3"
    >
      <h2 id="apply-title" className="text-sm font-bold text-text">
        Use these settings
      </h2>
      <p className="text-xs text-text-muted leading-snug">
        Loading replaces the model that is in memory now. Flash and image support take some memory,
        so the context can end up smaller than the numbers for the format alone.
      </p>
      {otherLoaded && (
        <p className="text-xs text-text-muted">
          {otherLoaded.dara.name} is loaded now and will be unloaded.
        </p>
      )}

      {isLoadingThis ? (
        <div role="status">
          <ProgressBar
            value={loadProgress?.percent ?? 0}
            label={`Loading${loadProgress?.stage ? `: ${loadProgress.stage}` : ""}`}
          />
        </div>
      ) : (
        <Button
          variant="primary"
          size="lg"
          disabled={!isInstalled || isBlocked || loadingModelId !== null}
          onClick={() => void apply()}
        >
          Load with these settings
        </Button>
      )}
      {!isInstalled && (
        <p className="text-xs text-text-muted">Install the model on the Models page to load it.</p>
      )}
      {failure && (
        <p role="alert" className="text-xs text-danger leading-snug">
          {failure}
        </p>
      )}

      {model?.dara.loaded ? (
        <LoadedConfigSummary model={model} />
      ) : (
        <p className="text-xs text-text-muted">
          {model ? `${model.dara.name} is not loaded.` : ""}
        </p>
      )}
    </section>
  );
};
