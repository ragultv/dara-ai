import React from "react";
import { useChatToolsStore } from "../../stores/useChatToolsStore.ts";

/** A small live line while tools run or wait for approval ("Running 2 tools"). Nothing when idle. */
export const ToolActivity: React.FC<{ conversationId: string }> = ({ conversationId }) => {
  const status = useChatToolsStore((s) => s.status[conversationId]);
  if (!status) return null;
  return (
    <div
      role="status"
      aria-live="polite"
      className="flex items-center gap-2 px-1 py-1 text-xs text-text-muted"
    >
      <span className="relative flex size-2" aria-hidden="true">
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-60" />
        <span className="relative inline-flex size-2 rounded-full bg-primary" />
      </span>
      <span>{status}</span>
    </div>
  );
};
