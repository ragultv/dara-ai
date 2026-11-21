import React, { useCallback, useEffect, useMemo } from "react";
import { useMonitorStore } from "../lib/stores/useMonitorStore.ts";
import { useSettingsStore } from "../lib/stores/useSettingsStore.ts";
import { formatDiagnostics } from "../lib/utils/redact.ts";
import { getAppVersion } from "../lib/utils/app-version.ts";
import { formatMb, formatTps } from "../lib/utils/format.ts";
import { Chart } from "../lib/ui/Chart.tsx";
import { Button } from "../lib/ui/Button.tsx";
import { Badge } from "../lib/ui/Badge.tsx";
import { ErrorState } from "../lib/ui/ErrorState.tsx";
import { Skeleton } from "../lib/ui/Skeleton.tsx";
import { Icon } from "../lib/ui/Icon.tsx";
import { toast } from "../lib/ui/Toast.tsx";
import { SummaryTiles } from "../lib/features/monitor/SummaryTiles.tsx";
import { GpuPanel } from "../lib/features/monitor/GpuPanel.tsx";
import { EnginePanel } from "../lib/features/monitor/EnginePanel.tsx";
import { MemoryPanel } from "../lib/features/monitor/MemoryPanel.tsx";
import { DiskPanel } from "../lib/features/monitor/DiskPanel.tsx";
import { DownloadsPanel } from "../lib/features/monitor/DownloadsPanel.tsx";
import { LastRequestPanel } from "../lib/features/monitor/LastRequestPanel.tsx";

const SLOTS = 120;

export const MonitorPage: React.FC = () => {
  const stats = useMonitorStore((s) => s.stats);
  const samples = useMonitorStore((s) => s.samples);
  const error = useMonitorStore((s) => s.error);
  const version = useMonitorStore((s) => s.version);
  const startPolling = useMonitorStore((s) => s.startPolling);
  const stopPolling = useMonitorStore((s) => s.stopPolling);
  const refresh = useMonitorStore((s) => s.refresh);
  const inference = useSettingsStore((s) => s.inference);

  // Polls once a second while this page is mounted and the window is visible; stops on leave.
  useEffect(() => {
    startPolling();
    return () => stopPolling();
  }, [startPolling, stopPolling]);

  const tpsPoints = useMemo(() => samples.map((s) => s.tps), [samples]);
  const vramPoints = useMemo(() => samples.map((s) => s.vramUsedMb), [samples]);
  const utilPoints = useMemo(() => samples.map((s) => s.gpuUtilPercent), [samples]);

  const copyDiagnostics = useCallback(async () => {
    if (!stats) return;
    const report = formatDiagnostics(stats, version ?? undefined, {
      appVersion: await getAppVersion(),
      config: {
        default_kv: inference.defaultKv,
        default_flash: inference.defaultFlash,
        vram_reserve_mb: inference.vramReserveMb,
        vision_default: inference.visionEnabled
      },
      errors: error ? [error] : undefined
    });
    try {
      await navigator.clipboard.writeText(report);
      toast.show(
        "Diagnostics copied. Paths are reduced to file names and no chats are included.",
        "success"
      );
    } catch {
      toast.show("The clipboard is not available.", "error");
    }
  }, [stats, version, inference, error]);

  const gpu = stats?.gpu && stats.gpu.nvml ? stats.gpu : undefined;
  const engine = stats?.engine;
  const liveTps =
    engine && engine.busy && engine.live_tokens_per_second > 0
      ? engine.live_tokens_per_second
      : null;
  const lastAverage =
    engine && engine.requests > 0 && engine.avg_decode_tokens_per_second > 0
      ? engine.avg_decode_tokens_per_second
      : null;
  const isLive = stats !== null && error === null;

  return (
    <div className="h-full overflow-y-auto bg-bg px-6 py-6 text-text custom-scrollbar">
      <div className="mx-auto max-w-6xl space-y-5 pb-12 text-left">
        <header className="flex flex-wrap items-center justify-between gap-3">
          <div className="space-y-0.5">
            <h1 className="text-xl font-bold tracking-tight">Monitor</h1>
            <p className="text-xs text-text-muted">
              Live readings from your GPU and the Dara engine, refreshed every second while this
              page is open.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant={isLive ? "success" : error ? "danger" : "neutral"} size="md">
              {isLive ? "Live" : error ? "Offline" : "Connecting"}
            </Badge>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => void copyDiagnostics()}
              disabled={!stats}
            >
              <Icon name="copy" size={13} />
              Copy diagnostics
            </Button>
          </div>
        </header>

        {error && !stats && (
          <ErrorState
            title="Cannot reach the engine"
            message={`${error}. Check that Dara is running, then try again.`}
            onRetry={() => void refresh()}
          />
        )}

        {error && stats && (
          <div
            role="alert"
            className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-danger/30 bg-danger-subtle px-4 py-2.5 text-xs text-danger"
          >
            <span>
              Lost contact with the engine ({error}). The numbers below are the last reading.
            </span>
            <Button variant="secondary" size="xs" onClick={() => void refresh()}>
              Try again
            </Button>
          </div>
        )}

        {!stats && !error && (
          <div className="space-y-3" aria-busy="true" aria-label="Loading readings">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-5">
              {Array.from({ length: 5 }, (_, i) => (
                <Skeleton key={i} className="h-20 w-full" />
              ))}
            </div>
            <Skeleton className="h-40 w-full" />
          </div>
        )}

        {stats && (
          <>
            <SummaryTiles stats={stats} />

            <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
              <Chart
                label="Decode speed"
                value={liveTps !== null ? formatTps(liveTps) : "Idle"}
                hint={
                  liveTps === null && lastAverage !== null
                    ? `last average ${formatTps(lastAverage)}`
                    : "last 2 minutes"
                }
                points={tpsPoints}
                slots={SLOTS}
                min={0}
                ariaLabel="Live decode speed in tokens per second over the last two minutes; gaps mean the engine was idle"
                emptyText="Appears while the engine generates"
              />
              {gpu && (
                <Chart
                  label="VRAM used"
                  value={`${formatMb(gpu.vram_used_mb)}`}
                  hint={`of ${formatMb(gpu.vram_total_mb)}`}
                  points={vramPoints}
                  slots={SLOTS}
                  min={0}
                  max={gpu.vram_total_mb}
                  ariaLabel="Video memory used over the last two minutes"
                />
              )}
              {gpu && gpu.utilization_percent !== undefined && (
                <Chart
                  label="GPU utilization"
                  value={`${Math.round(gpu.utilization_percent)}%`}
                  hint="last 2 minutes"
                  points={utilPoints}
                  slots={SLOTS}
                  min={0}
                  max={100}
                  ariaLabel="GPU utilization in percent over the last two minutes"
                />
              )}
            </div>

            <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
              {gpu && <GpuPanel gpu={gpu} />}
              <EnginePanel stats={stats} />
              <MemoryPanel stats={stats} />
              {stats.disk && <DiskPanel disk={stats.disk} />}
              <DownloadsPanel downloads={stats.downloads} />
            </div>

            <LastRequestPanel last={stats.last_request} />
          </>
        )}
      </div>
    </div>
  );
};
