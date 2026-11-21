import React from "react";
import type { Stats } from "../../api/types.ts";
import { formatGb, formatMb } from "../../utils/format.ts";
import { ProgressBar } from "../../ui/ProgressBar.tsx";
import { KeyValue } from "./KeyValue.tsx";
import { Panel } from "./Panel.tsx";

interface DiskPanelProps {
  disk: NonNullable<Stats["disk"]>;
}

export const DiskPanel: React.FC<DiskPanelProps> = ({ disk }) => {
  const usedPercent =
    disk.volume_total_gb > 0
      ? ((disk.volume_total_gb - disk.volume_free_gb) / disk.volume_total_gb) * 100
      : undefined;
  return (
    <Panel title="Disk" icon="hard-drive">
      {usedPercent !== undefined && (
        <ProgressBar
          label={`${formatGb(disk.volume_free_gb)} free of ${formatGb(disk.volume_total_gb)}`}
          value={usedPercent}
        />
      )}
      <KeyValue
        rows={[
          { label: "Models folder", value: disk.models_dir, title: disk.models_dir },
          { label: "Installed models", value: formatMb(disk.installed_models_mb) }
        ]}
      />
    </Panel>
  );
};
