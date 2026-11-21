import React, { useState } from "react";
import { Icon } from "../../ui/Icon.tsx";
import { humanizeToolName } from "../../connectors/catalog.ts";
import { extractBlocks } from "../../chat/tools/protocol.ts";
import { useChatToolsStore } from "../../stores/useChatToolsStore.ts";
import { useConnectorsStore } from "../../stores/useConnectorsStore.ts";
import type {
  CallDecision,
  CallState,
  CallStatus,
  ToolResultItem
} from "../../chat/tools/types.ts";
import type { ToolCallRecord } from "../../chat/tools/records.ts";

type RowStatus = CallStatus | "pending" | "invalid";

const STATUS: Record<RowStatus, { icon: string; label: string; cls: string; spin?: boolean }> = {
  pending: { icon: "clock", label: "Waiting", cls: "text-text-subtle" },
  waiting_approval: { icon: "hand", label: "Waiting for your approval", cls: "text-warning" },
  running: { icon: "refresh-cw", label: "Running", cls: "text-text-muted", spin: true },
  ok: { icon: "check", label: "Done", cls: "text-success" },
  error: { icon: "alert-triangle", label: "Failed", cls: "text-danger" },
  denied: { icon: "ban", label: "Not allowed", cls: "text-text-subtle" },
  skipped: { icon: "minus", label: "Stopped", cls: "text-text-subtle" },
  invalid: { icon: "alert-triangle", label: "Could not read this call", cls: "text-danger" }
};

/** How the call was let through or stopped, as shown on its row and in its details. */
export const DECISION_LABEL: Record<CallDecision, string> = {
  auto: "Allowed by your settings",
  approved_once: "Allowed once",
  approved_always: "Always allowed",
  denied_by_user: "Denied",
  blocked: "Blocked by your settings",
  skipped: "Stopped"
};
/** the decisions worth a word on the row itself (the others show in the details) */
const ROW_DECISIONS: ReadonlySet<CallDecision> = new Set([
  "approved_once",
  "approved_always",
  "denied_by_user"
]);

function formatMs(ms: number | undefined): string {
  if (ms == null) return "";
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}

function prettyArgs(args: Record<string, unknown>): string {
  try {
    return JSON.stringify(args, null, 2);
  } catch {
    return String(args);
  }
}

/** "gmail__search_messages" -> connector slug "gmail", tool "search_messages". */
function splitName(name: string): { slug: string; tool: string } {
  const i = name.indexOf("__");
  return i > 0 ? { slug: name.slice(0, i), tool: name.slice(i + 2) } : { slug: "", tool: name };
}

interface Row {
  id: string;
  name: string;
  connector: string;
  connectorId: string | null;
  tool: string;
  arguments: Record<string, unknown> | null;
  raw: string | null;
  status: RowStatus;
  output: string | undefined;
  durationMs: number | undefined;
  decision: CallDecision | undefined;
}

const CallRow: React.FC<{ row: Row }> = ({ row }) => {
  const [open, setOpen] = useState(false);
  const st = STATUS[row.status];
  const hasDetail = row.arguments !== null || row.raw !== null || !!row.output;
  const decisionText = row.decision ? DECISION_LABEL[row.decision] : "";
  const rowDecision = row.decision && ROW_DECISIONS.has(row.decision) ? decisionText : "";
  const tail =
    row.status === "ok"
      ? [rowDecision, formatMs(row.durationMs)].filter(Boolean).join(" · ")
      : row.status === "denied" && row.decision === "denied_by_user"
        ? "Denied"
        : rowDecision && row.status !== "waiting_approval" && row.status !== "running"
          ? `${st.label} · ${rowDecision}`
          : st.label;
  return (
    <li className="text-xs">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        disabled={!hasDetail}
        aria-expanded={open}
        aria-label={`${row.connector} ${row.tool}: ${st.label}${rowDecision ? `, ${rowDecision}` : ""}. ${open ? "Hide" : "Show"} details`}
        className="flex w-full items-center gap-2 rounded-md px-2 py-1 text-left hover:bg-surface-sunken cursor-pointer disabled:cursor-default"
      >
        <Icon
          name={st.icon}
          size={12}
          className={`shrink-0 ${st.cls} ${st.spin ? "animate-spin" : ""}`}
        />
        <span className="min-w-0 flex-1 truncate">
          <span className="text-text-muted">{row.connector}</span>
          <span className="text-text-subtle"> · </span>
          <span className="font-medium text-text">{row.tool}</span>
        </span>
        <span className="shrink-0 text-[11px] text-text-subtle tabular-nums">{tail}</span>
        {hasDetail && (
          <Icon
            name="chevron-right"
            size={12}
            className={`shrink-0 text-text-subtle transition-transform ${open ? "rotate-90" : ""}`}
          />
        )}
      </button>
      {open && (
        <div className="ml-6 mt-1 mb-1.5 space-y-1.5">
          {(decisionText || row.durationMs !== undefined) && (
            <p className="text-[11px] text-text-subtle">
              {[st.label, decisionText, formatMs(row.durationMs)].filter(Boolean).join(" · ")}
            </p>
          )}
          {row.raw !== null ? (
            <Detail label="What the model wrote" text={row.raw} />
          ) : (
            row.arguments && <Detail label="Arguments" text={prettyArgs(row.arguments)} />
          )}
          {row.output && <Detail label="Result" text={row.output} />}
        </div>
      )}
    </li>
  );
};

const Detail: React.FC<{ label: string; text: string }> = ({ label, text }) => (
  <div>
    <p className="text-[10px] font-bold uppercase tracking-wider text-text-subtle">{label}</p>
    <pre className="mt-0.5 max-h-60 overflow-auto rounded-md border border-border bg-surface-sunken p-2 text-[11px] leading-snug text-text-muted whitespace-pre-wrap break-words custom-scrollbar">
      {text}
    </pre>
  </div>
);

/**
 * The tool calls in one assistant reply: live state from the store while it runs; afterwards (and after a restart) what was
 * stored with the message (`records`), with the results from the next tool-results message.
 * Approvals are answered in ToolApprovalDialog; a waiting row only says so.
 */
export const ToolCalls: React.FC<{
  conversationId: string;
  assistantMessageId: string;
  content: string;
  results: ToolResultItem[] | null;
  records?: ToolCallRecord[] | undefined;
  streaming: boolean;
}> = ({ conversationId, assistantMessageId, content, results, records, streaming }) => {
  const liveCalls = useChatToolsStore((s) => s.calls);
  const entries = useConnectorsStore((s) => s.entries);

  const blocks = extractBlocks(content);
  if (blocks.length === 0) return null;

  const connectorName = (
    live: Pick<CallState, "connectorName" | "connectorId"> | undefined,
    slug: string
  ): string => {
    if (live?.connectorName) return live.connectorName;
    if (live?.connectorId) {
      const e = entries.find((x) => x.config.id === live.connectorId);
      if (e) return e.config.name;
    }
    return slug ? humanizeToolName(slug) : "Tool";
  };

  const rows: Row[] = blocks.map((b) => {
    const id = `${assistantMessageId}:${b.index}`;
    const live: (ToolCallRecord & { output?: string | undefined }) | undefined =
      liveCalls[id] ?? records?.find((r) => r.id === id);
    const result = results?.find((r) => r.id === id);
    if (!("name" in b)) {
      return {
        id,
        name: "",
        connector: "Tool",
        connectorId: null,
        tool: "Unreadable call",
        arguments: null,
        raw: b.raw,
        status: live?.status ?? result?.status ?? (streaming ? "pending" : "invalid"),
        output: live?.output ?? result?.text,
        durationMs: live?.durationMs,
        decision: live?.decision
      };
    }
    const { slug, tool } = splitName(b.name);
    const status: RowStatus =
      live?.status ?? result?.status ?? (streaming || !results ? "pending" : "skipped");
    return {
      id,
      name: b.name,
      connector: connectorName(live, slug),
      connectorId: live?.connectorId ?? null,
      tool: humanizeToolName(live?.tool ?? tool),
      arguments: b.arguments,
      raw: null,
      status,
      output: live?.output ?? result?.text,
      durationMs: live?.durationMs,
      decision: live?.decision
    };
  });

  return (
    <ul
      aria-label="Tools used"
      data-conversation={conversationId}
      className="my-2 rounded-lg border border-border bg-surface p-1"
    >
      {rows.map((row) => (
        <CallRow key={row.id} row={row} />
      ))}
    </ul>
  );
};
