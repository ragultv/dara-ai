import React, { useCallback, useEffect, useMemo, useState } from "react";
import { apiClient } from "../lib/api/client.ts";
import type { KvFormat, OptionsResponse } from "../lib/api/types.ts";
import { useRouter } from "../lib/router/useRouter.ts";
import { isDownloadActive, useModelsStore } from "../lib/stores/useModelsStore.ts";
import { Button } from "../lib/ui/Button.tsx";
import { ErrorState } from "../lib/ui/ErrorState.tsx";
import { Icon } from "../lib/ui/Icon.tsx";
import { Skeleton } from "../lib/ui/Skeleton.tsx";
import { Toggle } from "../lib/ui/Toggle.tsx";
import { toast } from "../lib/ui/Toast.tsx";
import { ApplyPanel } from "../lib/features/options/ApplyPanel.tsx";
import { BenchmarkPanel } from "../lib/features/options/BenchmarkPanel.tsx";
import { FormatSelector } from "../lib/features/options/FormatSelector.tsx";
import { ResultCard } from "../lib/features/options/ResultCard.tsx";
import {
  findOption,
  flashAvailability,
  initialSelection,
  visionAvailability
} from "../lib/features/options/optionsLogic.ts";
import { useBenchmark } from "../lib/features/options/useBenchmark.ts";
import { DownloadProgress } from "../lib/features/models/DownloadProgress.tsx";
import { diskCheck, missingDownloadMb, statusText } from "../lib/features/models/modelHelpers.ts";
import { formatMb } from "../lib/utils/format.ts";

interface ViewProps {
  modelId: string;
}

const OptionsView: React.FC<ViewProps> = ({ modelId }) => {
  const navigate = useRouter((s) => s.navigate);
  const model = useModelsStore((s) => s.models.find((m) => m.id === modelId));
  const hasModels = useModelsStore((s) => s.models.length > 0);
  const job = useModelsStore((s) => s.downloads[modelId]);
  const diskFreeGb = useModelsStore((s) => s.diskFreeGb);
  const loadingModelId = useModelsStore((s) => s.loadingModelId);
  const fetchModels = useModelsStore((s) => s.fetchModels);
  const downloadAddon = useModelsStore((s) => s.downloadAddon);

  const [data, setData] = useState<OptionsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reloadTick, setReloadTick] = useState(0);
  const [kvChoice, setKvChoice] = useState<KvFormat | null>(null);
  const [flashChoice, setFlashChoice] = useState<boolean | null>(null);
  const [visionChoice, setVisionChoice] = useState<boolean | null>(null);

  const flashAvail = flashAvailability(data, model);
  const visionAvail = visionAvailability(model);
  const defaults = useMemo(() => (data ? initialSelection(data, model) : null), [data, model]);
  const vision =
    (visionChoice ?? defaults?.vision ?? model?.dara.vision.installed ?? false) &&
    visionAvail.canEnable;
  const flash = (flashChoice ?? defaults?.flash ?? false) && flashAvail.canEnable;
  const kv: KvFormat = kvChoice ?? defaults?.kv ?? "q8";

  const flashInstalled = model?.dara.flash.installed;
  const visionInstalled = model?.dara.vision.installed;

  useEffect(() => {
    if (!hasModels) void fetchModels();
  }, [hasModels, fetchModels]);

  // Options depend on whether image support is on (it reserves memory), so a change refetches them.
  useEffect(() => {
    const controller = new AbortController();
    void (async () => {
      try {
        const next = await apiClient.getModelOptions(
          modelId,
          controller.signal,
          vision ? "" : "?vision=false"
        );
        setData(next);
        setError(null);
      } catch (err: unknown) {
        if (controller.signal.aborted) return;
        setError(err instanceof Error ? err.message : "Could not load the options.");
      }
    })();
    return () => controller.abort();
  }, [modelId, vision, reloadTick, flashInstalled, visionInstalled]);

  const onBenchmarkFinished = useCallback((): void => {
    setReloadTick((t) => t + 1);
    void fetchModels();
  }, [fetchModels]);
  const bench = useBenchmark(modelId, onBenchmarkFinished);

  const install = (kind: "mmproj" | "flash"): void => {
    downloadAddon(modelId, kind).catch((err: unknown) => {
      toast.show(err instanceof Error ? err.message : "Download failed", "error");
    });
  };

  if (error && !data) {
    return (
      <div className="p-6 max-w-4xl mx-auto">
        <ErrorState
          title="Could not load the options"
          message={error}
          onRetry={() => setReloadTick((t) => t + 1)}
        />
      </div>
    );
  }
  if (!data) {
    return (
      <div className="p-6 max-w-4xl mx-auto space-y-4" aria-busy="true">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-40 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  const option = findOption(data.options, kv, flash);
  const isRecommended = data.recommended.kv === kv && data.recommended.flash === flash;
  const battery = data.calibration.entries_measured_on_battery;
  const isBlocked = bench.isRunning || loadingModelId !== null;
  const name = model?.dara.name ?? data.name;
  const installing = isDownloadActive(job);
  const flashDisk = diskCheck(model ? missingDownloadMb(model, ["flash"]) : 0, diskFreeGb);
  const visionDisk = diskCheck(model ? missingDownloadMb(model, ["mmproj"]) : 0, diskFreeGb);

  return (
    <div className="p-6 max-w-4xl w-full mx-auto space-y-5 text-left select-text">
      <header className="space-y-1 border-b border-border pb-4">
        <h1 className="text-xl font-bold text-text">{name} options</h1>
        <p className="text-sm text-text-muted">
          For {data.machine.gpu} with{" "}
          <span className="font-mono tabular-nums">{formatMb(data.machine.vram_total_mb)}</span> of
          video memory.
        </p>
      </header>

      {model && model.dara.status !== "installed" && (
        <div
          role="status"
          className="rounded-xl bg-warning-subtle border border-border p-3 text-xs text-text flex flex-wrap items-center gap-3"
        >
          <span>
            {statusText(model.dara.status)}. You can look at the numbers, but you need to install{" "}
            {name} before you can load it.
          </span>
          <Button variant="secondary" size="sm" onClick={() => navigate("models")}>
            Go to Models
          </Button>
        </div>
      )}

      {battery > 0 && (
        <div
          role="alert"
          className="rounded-xl bg-warning-subtle border border-border p-3 text-xs text-text flex items-start gap-2"
        >
          <Icon name="alert-triangle" size={14} className="text-warning shrink-0 mt-0.5" />
          <span>
            {battery} of the saved speed measurements were taken while the laptop was on battery.
            Measure again while plugged in for numbers you can compare.
          </span>
        </div>
      )}

      <FormatSelector
        formats={data.formats}
        value={kv}
        recommendedKv={data.recommended.kv}
        disabled={isBlocked}
        onChange={setKvChoice}
      />

      <section
        aria-label="Flash and image support"
        className="rounded-2xl border border-border bg-surface p-4 space-y-4"
      >
        <div className="space-y-2">
          <Toggle
            label="Flash"
            checked={flash}
            disabled={!flashAvail.canEnable || isBlocked}
            onChange={setFlashChoice}
            description="Faster generation. It uses some memory, so the context gets smaller."
          />
          {flashAvail.reason && <p className="text-xs text-text-muted">{flashAvail.reason}</p>}
          {flashAvail.canInstall && !installing && (
            <Button
              variant="secondary"
              size="sm"
              disabled={!flashDisk.ok}
              onClick={() => install("flash")}
            >
              Install Flash drafter
            </Button>
          )}
          {flashAvail.canInstall && !flashDisk.ok && flashDisk.message && (
            <p role="alert" className="text-xs text-danger">
              {flashDisk.message}
            </p>
          )}
        </div>
        <div className="space-y-2">
          <Toggle
            label="Image support"
            checked={vision}
            disabled={!visionAvail.canEnable || isBlocked}
            onChange={setVisionChoice}
            description={data.vision.note || "Lets the model look at pictures you attach."}
          />
          {visionAvail.reason && <p className="text-xs text-text-muted">{visionAvail.reason}</p>}
          {visionAvail.canInstall && !installing && (
            <Button
              variant="secondary"
              size="sm"
              disabled={!visionDisk.ok}
              onClick={() => install("mmproj")}
            >
              Install image support
            </Button>
          )}
          {visionAvail.canInstall && !visionDisk.ok && visionDisk.message && (
            <p role="alert" className="text-xs text-danger">
              {visionDisk.message}
            </p>
          )}
        </div>
        {installing && job && <DownloadProgress job={job} title="Installing" />}
      </section>

      <ResultCard
        option={option}
        recommended={data.recommended}
        isRecommended={isRecommended}
        onUseRecommended={() => {
          setKvChoice(data.recommended.kv);
          setFlashChoice(data.recommended.flash);
        }}
      />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <BenchmarkPanel
          modelName={name}
          calibration={data.calibration}
          vision={vision}
          isBlocked={loadingModelId !== null}
          bench={bench}
        />
        <ApplyPanel
          modelId={modelId}
          model={model}
          kv={kv}
          flash={flash}
          vision={vision}
          isBlocked={bench.isRunning}
        />
      </div>

      {data.notes.length > 0 && (
        <ul className="text-xs text-text-muted list-disc pl-5 space-y-1">
          {data.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      )}
    </div>
  );
};

export const OptionsPage: React.FC = () => {
  const optionsModelId = useRouter((s) => s.optionsModelId);
  return <OptionsView key={optionsModelId} modelId={optionsModelId} />;
};
