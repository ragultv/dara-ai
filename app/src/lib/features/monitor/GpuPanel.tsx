import React from "react";
import type { Stats } from "../../api/types.ts";
import { formatMb } from "../../utils/format.ts";
import { Badge } from "../../ui/Badge.tsx";
import { ProgressBar } from "../../ui/ProgressBar.tsx";
import { KeyValue, type KeyValueRow } from "./KeyValue.tsx";
import { Panel } from "./Panel.tsx";

type Gpu = NonNullable<Stats["gpu"]>;

interface GpuPanelProps {
  gpu: Gpu;
}

/** Only rendered when the server reported NVML data; fields the server left out are not shown. */
export const GpuPanel: React.FC<GpuPanelProps> = ({ gpu }) => {
  const power =
    gpu.power_w !== undefined
      ? gpu.power_limit_w !== undefined
        ? `${gpu.power_w.toFixed(1)} W of ${gpu.power_limit_w.toFixed(0)} W`
        : `${gpu.power_w.toFixed(1)} W`
      : undefined;
  const rows: KeyValueRow[] = [
    {
      label: "Temperature",
      value: gpu.temperature_c !== undefined ? `${Math.round(gpu.temperature_c)} °C` : undefined
    },
    { label: "Power", value: power },
    {
      label: "Core clock (SM)",
      value: gpu.clock_sm_mhz !== undefined ? `${gpu.clock_sm_mhz} MHz` : undefined
    },
    {
      label: "Memory clock",
      value: gpu.clock_mem_mhz !== undefined ? `${gpu.clock_mem_mhz} MHz` : undefined
    },
    {
      label: "Performance state",
      value: gpu.pstate !== undefined ? `P${gpu.pstate}` : undefined,
      title: "P0 is the highest performance state"
    },
    { label: "Model weights in VRAM", value: formatMb(gpu.vram_weights_mb) },
    {
      label: "Memory bus busy",
      value:
        gpu.memory_utilization_percent !== undefined
          ? `${gpu.memory_utilization_percent}%`
          : undefined
    }
  ];

  return (
    <Panel
      title="GPU"
      icon="cpu"
      aside={
        <Badge variant="neutral" size="sm">
          {gpu.name}
          {gpu.compute_capability ? ` · sm ${gpu.compute_capability}` : ""}
        </Badge>
      }
    >
      <div className="space-y-1">
        <ProgressBar
          label={`VRAM  ${formatMb(gpu.vram_used_mb)} of ${formatMb(gpu.vram_total_mb)}`}
          value={gpu.vram_total_mb > 0 ? (gpu.vram_used_mb / gpu.vram_total_mb) * 100 : 0}
        />
        <p className="text-[11px] text-text-muted">{formatMb(gpu.vram_free_mb)} free</p>
      </div>
      {gpu.utilization_percent !== undefined && (
        <ProgressBar label="GPU utilization" value={gpu.utilization_percent} />
      )}
      <KeyValue rows={rows} />
    </Panel>
  );
};
