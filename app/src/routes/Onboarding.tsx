import React, { useEffect } from "react";
import { isDownloadActive, useModelsStore } from "../lib/stores/useModelsStore.ts";
import { useRouter } from "../lib/router/useRouter.ts";
import { Icon } from "../lib/ui/Icon.tsx";
import { Badge } from "../lib/ui/Badge.tsx";
import { Button } from "../lib/ui/Button.tsx";
import { Skeleton } from "../lib/ui/Skeleton.tsx";
import { ProgressBar } from "../lib/ui/ProgressBar.tsx";
import { StorageChooser } from "../lib/features/models/StorageChooser.tsx";
import { DownloadProgress } from "../lib/features/models/DownloadProgress.tsx";
import {
  diskCheck,
  fitsText,
  missingDownloadMb,
  recommendTier
} from "../lib/features/models/modelHelpers.ts";
import { formatContext, formatMb } from "../lib/utils/format.ts";
import { Logo } from "../lib/ui/Logo.tsx";

export const Onboarding: React.FC = () => {
  const models = useModelsStore((s) => s.models);
  const downloads = useModelsStore((s) => s.downloads);
  const diskFreeGb = useModelsStore((s) => s.diskFreeGb);
  const notice = useModelsStore((s) => s.notice);
  const error = useModelsStore((s) => s.error);
  const setupError = useModelsStore((s) => s.setupError);
  const setupModelId = useModelsStore((s) => s.setupModelId);
  const loadingModelId = useModelsStore((s) => s.loadingModelId);
  const loadProgress = useModelsStore((s) => s.loadProgress);
  const fetchModels = useModelsStore((s) => s.fetchModels);
  const setupModel = useModelsStore((s) => s.setupModel);
  const cancelDownload = useModelsStore((s) => s.cancelDownload);
  const navigate = useRouter((s) => s.navigate);

  useEffect(() => {
    void fetchModels();
  }, [fetchModels]);

  const recommendedId = recommendTier(models);
  const isSettingUp = setupModelId !== null;

  const start = (modelId: string): void => {
    // The flow lives in the store, so it survives this screen being replaced once the model is installed.
    void setupModel(modelId)
      .then(() => navigate("chat"))
      .catch(() => undefined);
  };

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="flex flex-col items-center p-8 max-w-5xl mx-auto space-y-8 text-left select-text">
        <header className="text-center space-y-3 max-w-xl">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-surface-raised border border-border mx-auto">
            <Logo alt="Dara" className="w-8 h-8 object-contain" />
          </div>
          <h1 className="text-3xl font-bold tracking-tight text-text">Welcome to Dara</h1>
          <p className="text-sm text-text-muted leading-relaxed">
            Download a model to start. Dara runs it on your own graphics card, so your chats stay on
            this computer. The recommended one fits your GPU.
          </p>
        </header>

        {!isSettingUp && <StorageChooser />}

        {(setupError || error) && (
          <div
            role="alert"
            className="w-full max-w-xl p-4 rounded-xl bg-danger-subtle border border-border text-xs text-danger font-medium flex items-center gap-2"
          >
            <Icon name="alert-triangle" size={16} />
            <span>{setupError ?? error}</span>
          </div>
        )}
        {notice && (
          <div
            role="status"
            className="w-full max-w-xl p-3 rounded-xl bg-warning-subtle border border-border text-xs text-text"
          >
            {notice}
          </div>
        )}

        {models.length === 0 && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-5 w-full" aria-busy="true">
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} className="h-72 w-full" />
            ))}
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5 w-full">
          {models.map((model) => {
            const d = model.dara;
            const job = downloads[model.id];
            const isDownloading = isDownloadActive(job);
            const isThisLoading = loadingModelId === model.id;
            const isThisSetup = setupModelId === model.id;
            const isRecommended = model.id === recommendedId;
            const fits = fitsText(d.fits_gpu);
            const needMb = missingDownloadMb(model, ["model", "mmproj"]);
            const disk = diskCheck(needMb, diskFreeGb);

            return (
              <article
                key={model.id}
                aria-labelledby={`onb-${model.id}`}
                className={`flex flex-col justify-between gap-4 bg-surface border rounded-2xl p-5 ${
                  isRecommended ? "border-border-strong " : "border-border"
                }`}
              >
                <div className="space-y-3">
                  <div className="flex items-center justify-between gap-2">
                    <h2 id={`onb-${model.id}`} className="text-base font-bold text-text">
                      {d.name}
                    </h2>
                    {isRecommended && (
                      <Badge variant="success" size="sm">
                        Recommended
                      </Badge>
                    )}
                  </div>
                  <p className="text-xs text-text-muted leading-relaxed">{d.description}</p>
                  <div className="text-xs space-y-1">
                    <Badge
                      variant={
                        d.fits_gpu === "full"
                          ? "success"
                          : d.fits_gpu === "partial"
                            ? "warning"
                            : "neutral"
                      }
                      size="sm"
                    >
                      {fits.label}
                    </Badge>
                    <p className="text-text-muted leading-snug">{fits.detail}</p>
                    <p className="text-text-muted">
                      Download:{" "}
                      <span className="font-mono">
                        {formatMb(needMb > 0 ? needMb : d.download_size_mb)}
                      </span>{" "}
                      (model and image support).
                      {typeof d.context === "number" && (
                        <>
                          {" "}
                          Up to <span className="font-mono">{formatContext(d.context)}</span>{" "}
                          tokens of context.
                        </>
                      )}
                    </p>
                  </div>
                </div>

                <div className="space-y-2">
                  {isDownloading && job ? (
                    <DownloadProgress
                      job={job}
                      title={`Downloading ${d.name}`}
                      onCancel={() => void cancelDownload(model.id)}
                    />
                  ) : isThisLoading ? (
                    <div role="status">
                      <ProgressBar
                        value={loadProgress?.percent ?? 0}
                        label={`Loading${loadProgress?.stage ? `: ${loadProgress.stage}` : ""}`}
                        size="sm"
                      />
                    </div>
                  ) : isThisSetup ? (
                    <p role="status" className="text-xs text-text-muted">
                      Getting {d.name} ready...
                    </p>
                  ) : (
                    <>
                      <Button
                        variant={isRecommended ? "primary" : "secondary"}
                        size="md"
                        className="w-full"
                        disabled={isSettingUp || !disk.ok}
                        onClick={() => start(model.id)}
                      >
                        Download and start
                      </Button>
                      {!disk.ok && disk.message && (
                        <p role="alert" className="text-xs text-danger">
                          {disk.message}
                        </p>
                      )}
                    </>
                  )}
                </div>
              </article>
            );
          })}
        </div>
      </div>
    </div>
  );
};
