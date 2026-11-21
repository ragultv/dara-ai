import React, { useCallback, useRef } from "react";
import { Dialog } from "../../ui/Dialog.tsx";
import { Button } from "../../ui/Button.tsx";
import { Icon } from "../../ui/Icon.tsx";
import { ConnectorAvatar } from "../../features/connectors/ConnectorAvatar.tsx";
import { humanizeToolName } from "../../connectors/catalog.ts";
import { useChatToolsStore } from "../../stores/useChatToolsStore.ts";
import { useConnectorsStore } from "../../stores/useConnectorsStore.ts";

/** One argument's value as text: plain for strings and numbers, pretty JSON for anything nested. */
function valueText(v: unknown): { text: string; block: boolean } {
  if (typeof v === "string") return { text: v, block: v.includes("\n") || v.length > 80 };
  if (v === null || typeof v === "number" || typeof v === "boolean")
    return { text: String(v), block: false };
  try {
    return { text: JSON.stringify(v, null, 2), block: true };
  } catch {
    return { text: String(v), block: false };
  }
}

/**
 * Asks before a tool call runs (docs/CHAT_TOOLS.md), one call at a time: the first call in the approval queue.
 * Mounted once in ChatPage. Escape or closing it denies; Stop resolves every waiting call, which empties the queue and closes it.
 */
export const ToolApprovalDialog: React.FC = () => {
  const approvals = useChatToolsStore((s) => s.approvals);
  const callId = approvals[0];
  const call = useChatToolsStore((s) => (callId ? s.calls[callId] : undefined));
  const resolve = useChatToolsStore((s) => s.resolveApproval);
  const entries = useConnectorsStore((s) => s.entries);
  const connectorTools = useConnectorsStore((s) =>
    call?.connectorId ? s.tools[call.connectorId] : undefined
  );
  const denyRef = useRef<HTMLButtonElement | null>(null);

  // a new function per call, so the dialog puts focus back on Deny when the next call comes up
  const deny = useCallback(() => {
    if (callId) resolve(callId, false);
  }, [callId, resolve]);

  if (!callId || !call) return null;

  const def = connectorTools?.find((t) => t.name === call.tool);
  const connector =
    call.connectorName ||
    entries.find((e) => e.config.id === call.connectorId)?.config.name ||
    "Connector";
  const toolTitle = def?.title?.trim() || humanizeToolName(call.tool);
  const args = Object.entries(call.arguments ?? {});

  return (
    <Dialog
      isOpen
      title={`Allow Dara to use ${toolTitle}?`}
      onclose={deny}
      initialFocus={denyRef}
      actions={
        <>
          <button
            ref={denyRef}
            type="button"
            onClick={deny}
            aria-label="Deny this call"
            className="inline-flex h-8 items-center justify-center rounded-xl px-3 text-xs font-medium text-text-muted hover:bg-surface hover:text-text transition-all cursor-pointer outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
          >
            Deny
          </button>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => resolve(callId, true, true)}
            aria-label={`Always allow ${toolTitle} in this chat`}
          >
            Always allow in this chat
          </Button>
          <Button
            variant="primary"
            size="sm"
            onClick={() => resolve(callId, true)}
            aria-label="Allow this call once"
          >
            Allow once
          </Button>
        </>
      }
    >
      <div className="flex items-center gap-3" data-testid="tool-approval">
        <ConnectorAvatar name={connector} />
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-semibold text-text">{connector}</p>
          <p className="truncate text-xs text-text-muted">{toolTitle}</p>
        </div>
        <span className="shrink-0 text-[11px] text-text-subtle tabular-nums">
          {`1 of ${approvals.length}`}
        </span>
      </div>

      {def?.description && (
        <p className="text-xs leading-relaxed text-text-muted whitespace-pre-wrap max-h-28 overflow-y-auto custom-scrollbar">
          {def.description}
        </p>
      )}

      <div>
        <p className="text-[10px] font-bold uppercase tracking-wider text-text-subtle">
          What it will send
        </p>
        {args.length === 0 ? (
          <p className="mt-1 text-xs text-text-subtle italic">No arguments</p>
        ) : (
          <dl className="mt-1 max-h-60 overflow-y-auto rounded-lg border border-border bg-surface-sunken divide-y divide-border custom-scrollbar">
            {args.map(([k, v]) => {
              const { text, block } = valueText(v);
              return (
                <div key={k} className={`px-2.5 py-1.5 ${block ? "" : "flex gap-3"}`}>
                  <dt className="shrink-0 text-[11px] font-semibold text-text-muted">
                    {humanizeToolName(k)}
                  </dt>
                  <dd className="min-w-0 text-[11px] text-text break-words">
                    {block ? (
                      <pre className="mt-0.5 whitespace-pre-wrap break-words font-mono leading-snug">
                        {text}
                      </pre>
                    ) : (
                      text
                    )}
                  </dd>
                </div>
              );
            })}
          </dl>
        )}
      </div>

      {def && def.readOnlyHint !== null && (
        <p className="flex items-start gap-1.5 text-[11px] leading-snug text-text-subtle">
          <Icon name="info" size={12} className="mt-px shrink-0" />
          <span>
            {def.readOnlyHint
              ? "The server says this tool only reads. Dara cannot check that, so it still asks."
              : "The server says this tool can make changes."}
          </span>
        </p>
      )}
    </Dialog>
  );
};
