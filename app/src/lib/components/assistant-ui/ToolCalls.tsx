import React, { useState } from "react";
import { Icon } from "../../ui/Icon.tsx";
import { humanizeToolName } from "../../connectors/catalog.ts";
import { displayText, extractBlocks } from "../../chat/tools/protocol.ts";
import { useChatToolsStore } from "../../stores/useChatToolsStore.ts";
import { useConnectorsStore } from "../../stores/useConnectorsStore.ts";
import { Markdown } from "../../ui/Markdown.tsx";
import { Reasoning } from "./Reasoning.tsx";
import type {
  CallDecision,
  CallState,
  CallStatus,
  ToolResultItem
} from "../../chat/tools/types.ts";
import type { ToolCallRecord } from "../../chat/tools/records.ts";
import type { ConnectorEntry } from "../../connectors/types.ts";
import type { UiMessage } from "../../chat/types.ts";

// Tool calls in a reply, laid out like assistant-ui's ToolCall element (https://www.assistant-ui.com/elements/tool-call): one line
// per call (chevron, label, the main argument as a chip, a check or a cross), opening to the request and the result. All the calls
// of one reply sit in one fold, like the thinking: open while tools run or wait for approval, folded to "Used Google Calendar" after.

type RowStatus = CallStatus | "pending" | "invalid";

/** How the call was let through or stopped. */
export const DECISION_LABEL: Record<CallDecision, string> = {
  auto: "Allowed by your settings",
  approved_once: "Allowed once",
  approved_always: "Always allowed",
  denied_by_user: "Denied",
  blocked: "Blocked by your settings",
  skipped: "Stopped"
};

const STATUS_TEXT: Record<RowStatus, string> = {
  pending: "Waiting",
  waiting_approval: "Waiting for your approval",
  running: "Running",
  ok: "Done",
  error: "Failed",
  denied: "Not allowed",
  skipped: "Stopped",
  invalid: "Could not be read"
};

export interface ToolRow {
  id: string;
  connector: string;
  tool: string;
  arguments: Record<string, unknown> | null;
  /** what the model wrote, for a call that could not be read */
  raw: string | null;
  status: RowStatus;
  output: string | undefined;
  durationMs: number | undefined;
  decision: CallDecision | undefined;
}

/** One step of a reply: an assistant message and the results message that answered its calls. */
export interface TurnStep {
  message: UiMessage;
  results: ToolResultItem[] | null;
}

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

/** The argument worth showing on the collapsed line: the first short text or number. */
function primaryArg(args: Record<string, unknown> | null): string {
  if (!args) return "";
  for (const v of Object.values(args)) {
    if (typeof v === "string" && v.trim()) return v.trim();
    if (typeof v === "number") return String(v);
  }
  return "";
}

/** "gmail__search_messages" -> connector slug "gmail", tool "search_messages". */
function splitName(name: string): { slug: string; tool: string } {
  const i = name.indexOf("__");
  return i > 0 ? { slug: name.slice(0, i), tool: name.slice(i + 2) } : { slug: "", tool: name };
}

/** The rows of one assistant message: live state while it runs; afterwards what was stored with the message, plus its results. */
export function stepRows(
  step: TurnStep,
  streaming: boolean,
  live: Record<string, CallState>,
  entries: ConnectorEntry[]
): ToolRow[] {
  const blocks = extractBlocks(step.message.content);
  const records = step.message.toolCalls;
  const rows: ToolRow[] = [];
  blocks.forEach((b, pos) => {
    const id = `${step.message.id}:${b.index}`;
    const st: (ToolCallRecord & { output?: string | undefined }) | undefined =
      live[id] ?? records?.find((r) => r.id === id);
    // a block that was not run because it was wrong is answered as "problem:<index>" (formatProblems): show it as failed, with why
    const result =
      step.results?.find((r) => r.id === id) ??
      step.results?.find((r) => r.id === `problem:${b.index}`);
    if (!("name" in b)) {
      // the last block of a reply still being written is not finished yet, not unreadable
      if (streaming && pos === blocks.length - 1) return;
      rows.push({
        id,
        connector: "",
        tool: "Unreadable tool call",
        arguments: null,
        raw: b.raw,
        status: st?.status ?? "invalid",
        output: st?.output ?? result?.text,
        durationMs: st?.durationMs,
        decision: st?.decision
      });
      return;
    }
    const { slug, tool } = splitName(b.name);
    const connectorName =
      st?.connectorName ||
      entries.find((e) => e.config.id === st?.connectorId)?.config.name ||
      (slug ? humanizeToolName(slug) : "");
    rows.push({
      id,
      connector: connectorName,
      tool: humanizeToolName(st?.tool ?? tool),
      arguments: b.arguments,
      raw: null,
      status: st?.status ?? result?.status ?? (streaming || !step.results ? "pending" : "skipped"),
      output: st?.output ?? result?.text,
      durationMs: st?.durationMs,
      decision: st?.decision
    });
  });
  return rows;
}

const StatusMark: React.FC<{ status: RowStatus }> = ({ status }) => {
  if (status === "running" || status === "pending")
    return (
      <span
        className="h-3 w-3 shrink-0 animate-spin rounded-full border-[1.5px] border-text-subtle border-t-transparent"
        aria-hidden="true"
      />
    );
  if (status === "waiting_approval")
    return <Icon name="hand" size={13} className="shrink-0 text-warning" />;
  if (status === "ok") return <Icon name="check" size={13} className="shrink-0 text-success" />;
  if (status === "skipped")
    return <Icon name="minus" size={13} className="shrink-0 text-text-subtle" />;
  if (status === "denied")
    return <Icon name="ban" size={13} className="shrink-0 text-text-subtle" />;
  return <Icon name="x" size={13} className="shrink-0 text-danger" />;
};

const Block: React.FC<{ label: string; text: string }> = ({ label, text }) => (
  <div>
    <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-text-subtle">
      {label}
    </p>
    <pre className="max-h-60 overflow-auto whitespace-pre-wrap break-words rounded-md bg-surface-sunken p-2 font-mono text-[11px] leading-snug text-text-muted custom-scrollbar">
      {text}
    </pre>
  </div>
);

/** One call: a single line that opens to the request and the result. */
export const ToolCall: React.FC<{ row: ToolRow }> = ({ row }) => {
  const [open, setOpen] = useState(false);
  const running = row.status === "running" || row.status === "pending";
  const query = primaryArg(row.arguments);
  const meta = [
    STATUS_TEXT[row.status],
    row.decision && row.decision !== "auto" && row.decision !== "skipped"
      ? DECISION_LABEL[row.decision]
      : "",
    formatMs(row.durationMs)
  ]
    .filter(Boolean)
    .join(" · ");
  return (
    <div data-slot="tool-call" className="text-[13px]">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="group/call flex w-full min-w-0 cursor-pointer items-center gap-2 rounded-md py-1 text-left text-text-muted transition-colors hover:text-text"
      >
        <Icon
          name="chevron-right"
          size={12}
          className={`shrink-0 text-text-subtle transition-transform duration-200 ${open ? "rotate-90" : ""}`}
        />
        <span className={`shrink-0 ${running ? "reasoning-shimmer" : ""}`}>
          {row.connector && <span className="text-text-subtle">{row.connector} · </span>}
          <span className="font-medium text-text">{row.tool}</span>
        </span>
        {query && (
          <span className="min-w-0 truncate rounded-md border border-border bg-surface-sunken px-1.5 py-px font-mono text-[11px] text-text-muted">
            {query}
          </span>
        )}
        <span className="ml-auto flex shrink-0 items-center gap-1.5 pl-2">
          {row.status === "waiting_approval" && (
            <span className="text-[11px] text-warning">Waiting for your approval</span>
          )}
          <StatusMark status={row.status} />
        </span>
      </button>
      <div
        className={`grid transition-[grid-template-rows] duration-200 ease-out ${open ? "grid-rows-[1fr]" : "grid-rows-[0fr]"}`}
      >
        <div className="min-h-0 overflow-hidden">
          {open && (
            <div className="mb-2 ml-5 space-y-2">
              <p className="text-[11px] text-text-subtle">{meta}</p>
              {row.raw !== null ? (
                <Block label="What the model wrote" text={row.raw} />
              ) : (
                <Block label="Request" text={row.arguments ? prettyArgs(row.arguments) : "{}"} />
              )}
              {row.output && (
                <>
                  <div className="h-px bg-border" />
                  <Block label="Result" text={row.output} />
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

/**
 * The thinking a tool step printed before its calls. Every step keeps its own, but only the last message of a reply showed it, so the
 * thinking between tool calls vanished once the answer was done. `skipReasoningOf` is the message the reply prints below the fold itself.
 */
export function stepThinking(
  step: TurnStep,
  skipReasoningOf?: string
): { text: string; durationMs: number | undefined; cut: boolean | undefined } | null {
  const text = step.message.reasoning?.trim();
  if (!text || step.message.id === skipReasoningOf) return null;
  return { text: step.message.reasoning ?? text, durationMs: step.message.reasoningMs, cut: step.message.reasoningCut };
}

function groupTitle(
  rows: ToolRow[],
  working: boolean
): { text: string; shimmer: boolean; tone: string } {
  const waiting = rows.find((r) => r.status === "waiting_approval");
  if (waiting)
    return {
      text: `Waiting for your approval: ${waiting.tool}`,
      shimmer: false,
      tone: "text-warning"
    };
  const running = rows.find((r) => r.status === "running" || r.status === "pending");
  if (running)
    return {
      text: `Using ${running.connector ? `${running.connector}: ` : ""}${running.tool}`,
      shimmer: true,
      tone: ""
    };
  const used = [...new Set(rows.filter((r) => r.connector).map((r) => r.connector))];
  const failed = rows.filter((r) => r.status === "error" || r.status === "invalid").length;
  const base = used.length > 0 ? `Used ${used.join(", ")}` : "Used tools";
  const count = rows.length === 1 ? "" : ` · ${rows.length} calls`;
  const fail = failed > 0 ? ` · ${failed} failed` : "";
  return { text: `${base}${count}${fail}`, shimmer: working, tone: "" };
}

/**
 * Every tool step of one reply in one fold. Inside, in order: what the model thought and said between steps (muted) and each call.
 * Open while a call runs or waits for approval, folded once done, unless the reader opened or closed it.
 */
export const ToolGroup: React.FC<{
  steps: TurnStep[];
  streaming: boolean;
  /** the message the reply is showing already prints its own thinking below the fold, so it is not repeated inside */
  skipReasoningOf?: string;
}> = ({ steps, streaming, skipReasoningOf }) => {
  const live = useChatToolsStore((s) => s.calls);
  const entries = useConnectorsStore((s) => s.entries);
  const [manual, setManual] = useState<boolean | null>(null);

  const parts = steps.map((step, i) => ({
    step,
    thinking: stepThinking(step, skipReasoningOf),
    text: displayText(step.message.content),
    rows: stepRows(step, streaming && i === steps.length - 1, live, entries)
  }));
  const rows = parts.flatMap((p) => p.rows);
  if (rows.length === 0) return null;
  const active = rows.some(
    (r) => r.status === "running" || r.status === "pending" || r.status === "waiting_approval"
  );
  const open = manual ?? active;
  const title = groupTitle(rows, streaming && !active);

  return (
    <div data-slot="tool-group" data-state={open ? "open" : "closed"} className="mb-2 text-left">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setManual(!open)}
        className={`flex h-7 max-w-full cursor-pointer items-center gap-2 rounded-md text-left text-[13px] text-text-muted transition-colors hover:text-text focus-visible:outline-2 focus-visible:outline-focus ${title.tone}`}
      >
        <Icon name="plug" size={13} className="shrink-0" />
        <span className={`truncate ${title.shimmer ? "reasoning-shimmer" : ""}`}>{title.text}</span>
        <Icon
          name="chevron-down"
          size={14}
          className={`shrink-0 transition-transform duration-200 ${open ? "rotate-180" : ""}`}
        />
      </button>
      <div
        className={`grid transition-[grid-template-rows] duration-200 ease-out ${open ? "grid-rows-[1fr]" : "grid-rows-[0fr]"}`}
      >
        <div className="min-h-0 overflow-hidden">
          <div className="mt-1 space-y-1 border-l border-border-strong/60 pl-4">
            {parts.map((p) => (
              <React.Fragment key={p.step.message.id}>
                {p.thinking && (
                  <Reasoning
                    text={p.thinking.text}
                    streaming={false}
                    durationMs={p.thinking.durationMs}
                    cut={p.thinking.cut}
                  />
                )}
                {p.text && (
                  <div className="reasoning-body py-0.5 text-xs leading-relaxed text-text-muted">
                    <Markdown content={p.text} />
                  </div>
                )}
                {p.rows.map((row) => (
                  <ToolCall key={row.id} row={row} />
                ))}
              </React.Fragment>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
