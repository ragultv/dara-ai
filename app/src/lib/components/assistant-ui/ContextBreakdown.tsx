import React from "react";
import { NumberTicker } from "../../ui/NumberTicker.tsx";
import { useRouter } from "../../router/useRouter.ts";
import { useModelsStore } from "../../stores/useModelsStore.ts";
import { getContextView } from "../../stores/useChatStore.ts";
import type { Conversation } from "../../chat/types.ts";

interface ContextBreakdownProps {
  conversation: Conversation | null;
  onClose?: () => void;
  className?: string;
}

/** What the context window is and how full it is. `used` is what the engine measured after the last reply (or an estimate marked "~"). */
export const ContextBreakdown: React.FC<ContextBreakdownProps> = ({
  conversation,
  onClose,
  className = ""
}) => {
  const loaded = useModelsStore((s) => s.models.find((m) => m.dara.loaded) ?? null);
  const navigate = useRouter((s) => s.navigate);
  const { used, capacity, estimated } = getContextView(conversation);
  const percent = capacity ? Math.min(100, Math.round((used / Math.max(1, capacity)) * 100)) : 0;
  const windows = conversation?.windows ?? [];
  const last = conversation?.context;
  const cfg = loaded?.dara.config;

  return (
    <div
      className={`w-80 rounded-2xl border border-border bg-surface-raised shadow-2xl p-4 text-left select-none ${className}`}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-bold text-text uppercase tracking-wider">Context window</span>
        <span className="text-xs font-mono font-bold text-primary tabular-nums">
          {estimated ? "~" : ""}
          <NumberTicker value={used} /> / {capacity ? capacity.toLocaleString() : "-"}
        </span>
      </div>

      <div
        className="h-2 w-full bg-surface-sunken rounded-full overflow-hidden mb-3 border border-border/50"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={capacity ?? 0}
        aria-valuenow={used}
        aria-label="Context window usage"
      >
        <div
          className="h-full bg-primary transition-all duration-300 rounded-full"
          style={{ width: `${percent}%` }}
        />
      </div>

      <dl className="space-y-1.5 pt-1 border-t border-border/40 text-[11px] text-text-muted">
        <div className="flex items-center justify-between">
          <dt>In use{estimated ? " (estimate until the next reply)" : ""}</dt>
          <dd className="font-mono text-text font-medium tabular-nums">
            {used.toLocaleString()} tokens
          </dd>
        </div>
        {capacity ? (
          <div className="flex items-center justify-between">
            <dt>Free</dt>
            <dd className="font-mono text-text font-medium tabular-nums">
              {Math.max(0, capacity - used).toLocaleString()} tokens
            </dd>
          </div>
        ) : null}
        <div className="flex items-center justify-between">
          <dt>Earlier windows</dt>
          <dd className="font-mono text-text font-medium">{windows.length}</dd>
        </div>
        {last?.compacted && (
          <div className="flex items-center justify-between text-warning">
            <dt>Engine trimmed history</dt>
            <dd className="font-mono tabular-nums">
              {last.summarized_messages} summarized, {last.dropped_messages} dropped
            </dd>
          </div>
        )}
      </dl>

      <p className="mt-3 pt-3 border-t border-border/40 text-[11px] leading-snug text-text-subtle">
        When the window fills, earlier messages stay saved and visible; a summary of them carries on
        in a new window.
        {cfg
          ? ` This model is loaded with ${cfg.context.toLocaleString()} tokens (K/V ${cfg.kv.toUpperCase()}${cfg.flash ? ", Flash on" : ""}${cfg.vision ? ", image support on" : ""}).`
          : ""}
      </p>
      {loaded && (
        <button
          type="button"
          onClick={() => {
            onClose?.();
            navigate("options", loaded.id);
          }}
          className="mt-2 text-[11px] font-semibold text-link hover:underline cursor-pointer"
        >
          Change the context size in Model options
        </button>
      )}
    </div>
  );
};
