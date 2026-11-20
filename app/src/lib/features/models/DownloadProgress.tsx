import React from "react";
import type { DownloadJob } from "../../api/types.ts";
import { formatEta, formatMb, formatRate } from "../../utils/format.ts";
import { ProgressBar } from "../../ui/ProgressBar.tsx";
import { Button } from "../../ui/Button.tsx";

interface Props {
  job: DownloadJob;
  title: string;
  onCancel?: (() => void) | undefined;
}

const BYTES_PER_MB = 1024 * 1024;

export const DownloadProgress: React.FC<Props> = ({ job, title, onCancel }) => {
  const percent = Math.min(100, Math.max(0, job.percent ?? 0));
  return (
    <div className="space-y-2" aria-live="off">
      <div
        role="progressbar"
        aria-label={title}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(percent)}
      >
        <ProgressBar value={percent} label={title} size="sm" />
      </div>
      <p className="text-[11px] text-text-muted font-mono tabular-nums">
        {formatMb(job.completed_bytes / BYTES_PER_MB)} of {formatMb(job.total_bytes / BYTES_PER_MB)}
        {job.mb_per_second > 0 ? ` · ${formatRate(job.mb_per_second)}` : ""}
        {job.eta_seconds > 0 ? ` · about ${formatEta(job.eta_seconds)} left` : ""}
        {job.files_total > 1
          ? ` · file ${Math.min(job.files_done + 1, job.files_total)} of ${job.files_total}`
          : ""}
      </p>
      {onCancel && (
        <>
          <Button variant="secondary" size="sm" className="w-full" onClick={onCancel}>
            Cancel download
          </Button>
          <p className="text-[11px] text-text-muted">
            Cancelling keeps the part already downloaded, so you can resume later.
          </p>
        </>
      )}
    </div>
  );
};
