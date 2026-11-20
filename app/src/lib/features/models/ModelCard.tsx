import React from "react";
import type { DownloadJob, LoadStatus, ModelInfo } from "../../api/types.ts";
import { formatContext, formatMb, formatTokensExact } from "../../utils/format.ts";
import { Badge } from "../../ui/Badge.tsx";
import { Button } from "../../ui/Button.tsx";
import { ProgressBar } from "../../ui/ProgressBar.tsx";
import { isDownloadActive, type AddonKind } from "../../stores/useModelsStore.ts";
import { DownloadProgress } from "./DownloadProgress.tsx";
import { LoadedConfigSummary, ModelFiles } from "./ModelFiles.tsx";
import {
  diskCheck,
  fileRows,
  fitsText,
  installedFiles,
  missingDownloadMb
} from "./modelHelpers.ts";

interface Props {
  model: ModelInfo;
  job: DownloadJob | undefined;
  diskFreeGb: number | null;
  isLoadingThis: boolean;
  loadProgress: LoadStatus | null;
  isBusy: boolean;
  onDownload: () => void;
  onAddon: (kind: AddonKind) => void;
  onCancel: () => void;
  onLoad: () => void;
  onUnload: () => void;
  onDelete: () => void;
  onOptions: () => void;
  onChat: () => void;
}

export const ModelCard: React.FC<Props> = (props) => {
  const { model, job, diskFreeGb, isLoadingThis, loadProgress, isBusy } = props;
  const d = model.dara;
  const fits = fitsText(d.fits_gpu);
  const isInstalled = d.status === "installed";
  const isDownloading = isDownloadActive(job);
  const hasFiles = installedFiles(model).length > 0;
  const needMb = missingDownloadMb(model, ["model", "mmproj"]);
  const disk = diskCheck(needMb, diskFreeGb);
  const rows = fileRows(model);
  const mmprojMissing = rows.find((r) => r.kind === "mmproj")?.state === "available";
  const flashMissing = rows.find((r) => r.kind === "flash")?.state === "available";
  const flashNote = !d.flash.available ? "Flash: not available for this model" : null;

  return (
    <article
      aria-labelledby={`model-${model.id}`}
      className={`flex flex-col gap-4 bg-surface border rounded-md p-5 ${
        d.loaded ? "border-border-strong " : "border-border"
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div>
          <h2 id={`model-${model.id}`} className="text-base font-bold text-text tracking-tight">
            {d.name}
          </h2>
          <p className="text-xs text-text-muted mt-0.5">
            {d.base_model} · {d.quant}
          </p>
        </div>
        {d.loaded && (
          <Badge variant="primary" size="sm">
            Loaded
          </Badge>
        )}
        {!d.loaded && isInstalled && (
          <Badge variant="success" size="sm">
            Installed
          </Badge>
        )}
        {!d.loaded && d.status === "partial" && (
          <Badge variant="warning" size="sm">
            Some files missing
          </Badge>
        )}
      </div>

      <p className="text-xs text-text-muted leading-relaxed">{d.description}</p>

      <div className="text-xs space-y-1">
        <Badge
          variant={
            d.fits_gpu === "full" ? "success" : d.fits_gpu === "partial" ? "warning" : "neutral"
          }
          size="sm"
        >
          {fits.label}
        </Badge>
        <p className="text-text-muted leading-snug">{fits.detail}</p>
        <p className="text-text-muted">
          {typeof d.context === "number" && (
            <>
              Up to{" "}
              <span title={formatTokensExact(d.context)} className="font-mono tabular-nums">
                {formatContext(d.context)}
              </span>{" "}
              tokens of context.{" "}
            </>
          )}
          Full download: <span className="font-mono">{formatMb(d.download_size_mb)}</span>.
        </p>
      </div>

      <ModelFiles model={model} />
      {flashNote && <p className="text-xs text-text-muted">{flashNote}</p>}
      <LoadedConfigSummary model={model} />

      <div className="mt-auto pt-4 border-t border-border space-y-2">
        {isDownloading && job ? (
          <DownloadProgress job={job} title={`Downloading ${d.name}`} onCancel={props.onCancel} />
        ) : isLoadingThis ? (
          <div role="status" className="space-y-1">
            <ProgressBar
              value={loadProgress?.percent ?? 0}
              label={`Loading${loadProgress?.stage ? `: ${loadProgress.stage}` : ""}`}
              size="sm"
            />
          </div>
        ) : (
          <>
            {!isInstalled && (
              <Button
                variant="primary"
                size="sm"
                className="w-full"
                disabled={!disk.ok}
                onClick={props.onDownload}
              >
                {d.status === "partial" ? "Download missing files" : "Download"} ({formatMb(needMb)}
                )
              </Button>
            )}
            {!disk.ok && disk.message && (
              <p role="alert" className="text-xs text-danger">
                {disk.message}
              </p>
            )}
            {isInstalled && (
              <div className="flex items-center gap-2">
                {d.loaded ? (
                  <>
                    <Button variant="primary" size="sm" className="flex-1" onClick={props.onChat}>
                      Open chat
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      disabled={isBusy}
                      onClick={props.onUnload}
                    >
                      Unload
                    </Button>
                  </>
                ) : (
                  <Button
                    variant="primary"
                    size="sm"
                    className="flex-1"
                    disabled={isBusy}
                    onClick={props.onLoad}
                  >
                    Load
                  </Button>
                )}
                <Button variant="secondary" size="sm" onClick={props.onOptions}>
                  Options
                </Button>
              </div>
            )}
            {isInstalled && (mmprojMissing || flashMissing) && (
              <div className="flex flex-wrap gap-2">
                {mmprojMissing && (
                  <Button
                    variant="secondary"
                    size="sm"
                    disabled={!diskCheck(missingDownloadMb(model, ["mmproj"]), diskFreeGb).ok}
                    onClick={() => props.onAddon("mmproj")}
                  >
                    Add image support
                  </Button>
                )}
                {flashMissing && (
                  <Button
                    variant="secondary"
                    size="sm"
                    disabled={!diskCheck(missingDownloadMb(model, ["flash"]), diskFreeGb).ok}
                    onClick={() => props.onAddon("flash")}
                  >
                    Add Flash
                  </Button>
                )}
              </div>
            )}
            {hasFiles && (
              <Button
                variant="ghost"
                size="sm"
                className="w-full"
                disabled={isBusy || d.loaded}
                onClick={props.onDelete}
              >
                Delete files
              </Button>
            )}
            {hasFiles && d.loaded && (
              <p className="text-xs text-text-muted">Unload the model before deleting its files.</p>
            )}
          </>
        )}
      </div>
    </article>
  );
};
