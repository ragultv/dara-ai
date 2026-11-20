import React, { useState } from "react";
import { Icon } from "../../ui/Icon.tsx";
import { toast } from "../../ui/Toast.tsx";
import type { MessageStats } from "../../chat/types.ts";

export interface MessageActionsProps {
  content: string;
  stats?: MessageStats | undefined;
  onRegenerate?: (() => void) | undefined;
  className?: string | undefined;
}

function fmtMs(ms: number | undefined): string | undefined {
  if (ms === undefined || !Number.isFinite(ms)) return undefined;
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}

/** Copy / regenerate plus the numbers the engine reported for this reply. Nothing is shown that the engine did not measure. */
export const MessageActions: React.FC<MessageActionsProps> = ({
  content,
  stats,
  onRegenerate,
  className = ""
}) => {
  const [copied, setCopied] = useState(false);
  const spec = stats?.speculative;
  const flashWorked =
    spec && spec.mode === "flash" && spec.drafted > 0 && spec.flash_stood_down_at_pass === null;

  const handleCopy = () => {
    void navigator.clipboard.writeText(content).then(
      () => {
        setCopied(true);
        toast.show("Copied to clipboard.", "info");
        setTimeout(() => setCopied(false), 2000);
      },
      () => toast.show("Clipboard access was denied.", "error")
    );
  };

  const items: Array<{ key: string; text: string; title: string }> = [];
  if (stats?.decodeTokens !== undefined)
    items.push({
      key: "tok",
      text: `${stats.decodeTokens.toLocaleString()} tokens`,
      title: "Tokens in this reply"
    });
  const ttft = fmtMs(stats?.ttftMs);
  if (ttft) items.push({ key: "ttft", text: `first token ${ttft}`, title: "Time to first token" });
  if (stats?.cachedTokens)
    items.push({
      key: "cache",
      text: `${stats.cachedTokens.toLocaleString()} cached`,
      title: "Prompt tokens reused from the engine's cache"
    });
  if (spec && spec.mode === "flash") {
    items.push({
      key: "flash",
      text: flashWorked
        ? `Flash ${spec.tokens_per_pass.toFixed(2)} tok/pass`
        : spec.flash_stood_down_at_pass !== null
          ? "Flash paused"
          : "Flash",
      title:
        spec.note ??
        `Flash accepted ${spec.accepted} of ${spec.drafted} drafted tokens (${spec.acceptance}%)`
    });
  }

  return (
    <div
      className={`flex items-center justify-between w-full pt-1.5 text-xs text-text-subtle select-none ${className}`}
      role="toolbar"
      aria-label="Message actions"
    >
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 font-mono text-[11px] tabular-nums">
        {stats?.decodeTokensPerSecond ? (
          <span className="inline-flex items-center gap-1 font-semibold text-success bg-success/10 border border-success/20 px-2 py-0.5 rounded-full">
            <Icon name="zap" size={11} />
            {stats.decodeTokensPerSecond.toFixed(1)} tok/s
          </span>
        ) : null}
        {items.map((i) => (
          <span key={i.key} title={i.title} className="text-text-subtle">
            {i.text}
          </span>
        ))}
      </div>

      <div className="flex items-center gap-1">
        <button
          type="button"
          aria-label="Copy response"
          onClick={handleCopy}
          className="w-7 h-7 rounded-md inline-flex items-center justify-center text-text-subtle hover:text-text hover:bg-surface-raised transition-colors cursor-pointer"
          title={copied ? "Copied!" : "Copy response"}
        >
          {copied ? (
            <Icon name="check" size={13} className="text-success" />
          ) : (
            <Icon name="copy" size={13} />
          )}
        </button>
        {onRegenerate && (
          <button
            type="button"
            aria-label="Regenerate response"
            onClick={onRegenerate}
            className="w-7 h-7 rounded-md inline-flex items-center justify-center text-text-subtle hover:text-text hover:bg-surface-raised transition-colors cursor-pointer"
            title="Regenerate response"
          >
            <Icon name="refresh-cw" size={12} />
          </button>
        )}
      </div>
    </div>
  );
};
