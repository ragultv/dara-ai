import React from "react";
import type { Stats } from "../../api/types.ts";
import { formatMb } from "../../utils/format.ts";
import { ProgressBar } from "../../ui/ProgressBar.tsx";
import { KeyValue, type KeyValueRow } from "./KeyValue.tsx";
import { Panel } from "./Panel.tsx";

interface MemoryPanelProps {
  stats: Stats;
}

function mb(record: Record<string, number> | undefined, key: string): string | undefined {
  const v = record?.[key];
  return typeof v === "number" && Number.isFinite(v) ? formatMb(v) : undefined;
}

/** Process and system RAM as the server reports them (every value is megabytes in the API). */
export const MemoryPanel: React.FC<MemoryPanelProps> = ({ stats }) => {
  const { process, ram } = stats;
  if (!process && !ram) return null;

  const processRows: KeyValueRow[] = [
    { label: "In use (working set)", value: mb(process, "working_set_mb") },
    { label: "Private", value: mb(process, "private_working_set_mb") },
    { label: "Peak", value: mb(process, "peak_working_set_mb") },
    {
      label: "Committed",
      value: mb(process, "committed_mb"),
      title: "Memory the engine has reserved with Windows; it can be larger than what is in use"
    }
  ];
  const usedMb = ram?.["used_mb"];
  const totalMb = ram?.["total_mb"];
  const usedPercent = ram?.["used_percent"];

  return (
    <Panel title="Memory (RAM)" icon="memory-stick">
      {process && (
        <div className="space-y-2">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-text-muted">
            Dara engine process
          </p>
          <KeyValue rows={processRows} />
        </div>
      )}
      {ram && typeof usedMb === "number" && typeof totalMb === "number" && totalMb > 0 && (
        <div className="space-y-1">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-text-muted">
            Whole computer
          </p>
          <ProgressBar
            label={`${formatMb(usedMb)} of ${formatMb(totalMb)} used`}
            value={typeof usedPercent === "number" ? usedPercent : (usedMb / totalMb) * 100}
          />
          <KeyValue rows={[{ label: "Available", value: mb(ram, "available_mb") }]} />
        </div>
      )}
    </Panel>
  );
};
