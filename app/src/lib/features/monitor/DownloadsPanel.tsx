import React from "react";
import type { DownloadJob } from "../../api/types.ts";
import { formatEta, formatRate } from "../../utils/format.ts";
import { ProgressBar } from "../../ui/ProgressBar.tsx";
import { Panel } from "./Panel.tsx";

interface DownloadsPanelProps {
  downloads: DownloadJob[];
}

export const DownloadsPanel: React.FC<DownloadsPanelProps> = ({ downloads }) => (
  <Panel title="Downloads" icon="download">
    {downloads.length === 0 ? (
      <p className="text-xs text-text-muted">No downloads are running.</p>
    ) : (
      downloads.map((d) => (
        <div key={d.id} className="space-y-1">
          <ProgressBar label={`${d.model} · ${d.file}`} value={d.percent} />
          <p className="text-[11px] tabular-nums text-text-muted">
            {d.status} · file {d.files_done} of {d.files_total} · {formatRate(d.mb_per_second)} ·{" "}
            {formatEta(d.eta_seconds)} left
            {d.error ? ` · ${d.error}` : ""}
          </p>
        </div>
      ))
    )}
  </Panel>
);
