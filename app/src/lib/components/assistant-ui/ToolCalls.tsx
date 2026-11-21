import React, { useState } from "react";
import { Icon } from "../../ui/Icon.tsx";
import { Button } from "../../ui/Button.tsx";
import { humanizeToolName } from "../../connectors/catalog.ts";
import { extractBlocks } from "../../chat/tools/protocol.ts";
import { useChatToolsStore } from "../../stores/useChatToolsStore.ts";
import { useConnectorsStore } from "../../stores/useConnectorsStore.ts";
import type { CallState, CallStatus, ToolResultItem } from "../../chat/tools/types.ts";

type RowStatus = CallStatus | "pending" | "invalid";

const STATUS: Record<RowStatus, { icon: string; label: string; cls: string; spin?: boolean }> = {
  pending: { icon: "clock", label: "Waiting", cls: "text-text-subtle" },
  waiting_approval: { icon: "hand", label: "Needs your approval", cls: "text-warning" },
  running: { icon: "refresh-cw", label: "Running", cls: "text-text-muted", spin: true },
  ok: { icon: "check", label: "Done", cls: "text-success" },
  error: { icon: "alert-triangle", label: "Failed", cls: "text-danger" },
  denied: { icon: "ban", label: "Not allowed", cls: "text-text-subtle" },
  skipped: { icon: "minus", label: "Stopped", cls: "text-text-subtle" },
  invalid: { icon: "alert-triangle", label: "Could not read this call", cls: "text-danger" }
};

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
}

/** The approval card shown inline while a call waits for the user. */
const ApprovalCard: React.FC<{ row: Row }> = ({ row }) => {
  const resolve = useChatToolsStore((s) => s.resolveApproval);
  const args = row.arguments ? prettyArgs(row.arguments) : "";
  return (
    <div className="mt-1.5 rounded-lg border border-border bg-surface-raised p-3">
      <p className="text-xs text-text">
        Dara wants to use <span className="font-semibold">{row.tool}</span> from{" "}
        <span className="font-semibold">{row.connector}</span>
      </p>
      {args && args !== "{}" && (
        <pre className="mt-2 max-h-40 overflow-auto rounded-md bg-surface-sunken p-2 text-[11px] leading-snug text-text-muted whitespace-pre-wrap break-words custom-scrollbar">
          {args}
        </pre>
      )}
      <div className="mt-2.5 flex flex-wrap gap-1.5">
        <Button size="xs" variant="primary" onClick={() => resolve(row.id, true)}>
          Allow once
        </Button>
        <Button size="xs" variant="secondary" onClick={() => resolve(row.id, true, true)}>
          Always allow in this chat
        </Button>
        <Button size="xs" variant="ghost" onClick={() => resolve(row.id, false)}>
          Deny
        </Button>
      </div>
    </div>
  );
};

const CallRow: React.FC<{ row: Row; waiting: boolean }> = ({ row, waiting }) => {
  const [open, setOpen] = useState(false);
  const st = STATUS[row.status];
  const hasDetail = row.arguments !== null || row.raw !== null || !!row.output;
  return (
    <li className="text-xs">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        disabled={!hasDetail}
        aria-expanded={open}
        aria-label={`${row.connector} ${row.tool}: ${st.label}. ${open ? "Hide" : "Show"} details`}
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
        <span className="shrink-0 text-[11px] text-text-subtle tabular-nums">
          {row.status === "ok" ? formatMs(row.durationMs) : st.label}
        </span>
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
          {row.raw !== null ? (
            <Detail label="What the model wrote" text={row.raw} />
          ) : (
            row.arguments && <Detail label="Arguments" text={prettyArgs(row.arguments)} />
          )}
          {row.output && <Detail label="Result" text={row.output} />}
        </div>
      )}
      {waiting && <ApprovalCard row={row} />}
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

/** The tool calls in one assistant reply: live state from the store, finished results from the next tool-results message. */
export const ToolCalls: React.FC<{
  conversationId: string;
  assistantMessageId: string;
  content: string;
  results: ToolResultItem[] | null;
  streaming: boolean;
}> = ({ conversationId, assistantMessageId, content, results, streaming }) => {
  const calls = useChatToolsStore((s) => s.calls);
  const approvals = useChatToolsStore((s) => s.approvals);
  const entries = useConnectorsStore((s) => s.entries);

  const blocks = extractBlocks(content);
  if (blocks.length === 0) return null;

  const connectorName = (live: CallState | undefined, slug: string): string => {
    if (live?.connectorName) return live.connectorName;
    if (live?.connectorId) {
      const e = entries.find((x) => x.config.id === live.connectorId);
      if (e) return e.config.name;
    }
    return slug ? humanizeToolName(slug) : "Tool";
  };

  const rows: Row[] = blocks.map((b) => {
    const id = `${assistantMessageId}:${b.index}`;
    const live = calls[id];
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
        durationMs: live?.durationMs
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
      durationMs: live?.durationMs
    };
  });

  return (
    <ul
      aria-label="Tools used"
      data-conversation={conversationId}
      className="my-2 rounded-lg border border-border bg-surface p-1"
    >
      {rows.map((row) => (
        <CallRow
          key={row.id}
          row={row}
          waiting={row.status === "waiting_approval" && approvals.includes(row.id)}
        />
      ))}
    </ul>
  );
};
