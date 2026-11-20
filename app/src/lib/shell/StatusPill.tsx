import React from "react";
import { useServerStore } from "../stores/useServerStore.ts";
import { useChatStore } from "../stores/useChatStore.ts";
import { Icon } from "../ui/Icon.tsx";

export interface StatusPillProps {
  className?: string;
}

export const StatusPill: React.FC<StatusPillProps> = ({ className = "" }) => {
  const isReady = useServerStore((s) => s.isReady);
  const health = useServerStore((s) => s.health);
  const loadedModel = useServerStore((s) => s.loadedModel);
  const currentTps = useChatStore((s) => s.currentTps);

  const statusColor = isReady
    ? "bg-success"
    : health?.status === "loading"
      ? "bg-warning animate-pulse"
      : "bg-danger";

  const statusLabel = isReady
    ? "Engine ready"
    : health?.status === "loading"
      ? "Starting Engine"
      : "Engine Offline";

  return (
    <footer
      className={`h-9 px-4 border-t border-border bg-bg-subtle flex items-center justify-between text-xs text-text-muted select-none shrink-0 ${className}`}
      role="status"
      aria-label="Application Status"
    >
      {/* Left: Engine Status & Loaded Model */}
      <div className="flex items-center space-x-3">
        <div className="flex items-center space-x-2" title={statusLabel}>
          <span className={`w-2 h-2 rounded-full ${statusColor}`} />
          <span className="font-medium text-text">{statusLabel}</span>
          {isReady && (
            <span className="inline-flex items-center px-1.5 py-0.5 rounded-full text-[9px] font-bold bg-success/15 text-success border border-success/20">
              LIVE
            </span>
          )}
        </div>

        {loadedModel && (
          <div className="hidden sm:flex items-center space-x-1.5 text-text-subtle border-l border-border pl-3">
            <Icon name="cpu" size={13} />
            <span className="font-mono text-[11px] truncate max-w-[180px]">{loadedModel}</span>
          </div>
        )}
      </div>

      {/* Right: Telemetry pill */}
      <div className="flex items-center space-x-3 font-mono text-[11px]">
        {currentTps !== undefined && currentTps > 0 && (
          <div className="flex items-center gap-1.5 text-success font-semibold bg-success/10 px-2 py-0.5 rounded-full border border-success/20">
            <Icon name="zap" size={12} />
            <span>{currentTps.toFixed(1)} tok/s</span>
          </div>
        )}
      </div>
    </footer>
  );
};
