import React, { useEffect, useRef, useState } from "react";
import { useConnectorsStore } from "../lib/stores/useConnectorsStore.ts";
import { getBackend } from "../lib/connectors/backend.ts";
import { AddConnectorDialog } from "../lib/connectors/AddConnectorDialog.tsx";
import { ToolTestDialog } from "../lib/connectors/ToolTestDialog.tsx";
import type { ConnectorEntry, ConnectorTool, ToolPermission } from "../lib/connectors/types.ts";
import { relativeTime } from "../lib/artifacts/format.ts";
import { Icon } from "../lib/ui/Icon.tsx";
import { Button } from "../lib/ui/Button.tsx";
import { Dialog } from "../lib/ui/Dialog.tsx";
import { Skeleton } from "../lib/ui/Skeleton.tsx";

const PERMISSIONS: Array<{ value: ToolPermission; label: string; hint: string }> = [
  { value: "ask", label: "Ask", hint: "Dara asks every time" },
  { value: "allow", label: "Allow", hint: "Dara may use it without asking" },
  { value: "deny", label: "Block", hint: "Dara can never use it" }
];

const STATE_TEXT = {
  disconnected: "Not connected",
  connecting: "Connecting...",
  ready: "Connected",
  needs_auth: "Needs sign-in",
  error: "Could not connect"
} as const;

const StateDot: React.FC<{ state: keyof typeof STATE_TEXT }> = ({ state }) => (
  <span
    aria-hidden="true"
    className={`inline-block size-2 rounded-full ${
      state === "ready"
        ? "bg-success"
        : state === "error"
          ? "bg-danger"
          : state === "needs_auth"
            ? "bg-warning"
            : state === "connecting"
              ? "animate-pulse bg-text-muted"
              : "bg-border-strong"
    }`}
  />
);

const ToolRow: React.FC<{
  tool: ConnectorTool;
  onPermission: (p: ToolPermission) => void;
  onTry: () => void;
}> = ({ tool, onPermission, onTry }) => (
  <li
    className={`flex flex-col gap-2 py-3 sm:flex-row sm:items-start ${tool.available ? "" : "opacity-60"}`}
  >
    <div className="min-w-0 flex-1">
      <p className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-xs font-semibold text-text">{tool.name}</span>
        {tool.readOnlyHint === true && (
          <span
            className="rounded-full border border-border px-1.5 py-px text-[10px] text-text-subtle"
            title="The server says this tool only reads. Dara does not rely on that: you decide."
          >
            says read-only
          </span>
        )}
        {tool.changed && (
          <span
            className="rounded-full border border-warning/40 bg-warning-subtle px-1.5 py-px text-[10px] font-medium text-warning"
            title="This tool changed since you last chose a permission, so it was put back to Ask."
          >
            changed
          </span>
        )}
        {!tool.available && (
          <span className="rounded-full border border-border px-1.5 py-px text-[10px] text-text-subtle">
            no longer offered
          </span>
        )}
      </p>
      {tool.description && (
        <p className="mt-0.5 line-clamp-2 text-xs text-text-muted">{tool.description}</p>
      )}
    </div>
    <div className="flex shrink-0 items-center gap-2">
      <div
        role="group"
        aria-label={`Permission for ${tool.name}`}
        className="flex rounded-lg border border-border p-0.5"
      >
        {PERMISSIONS.map((p) => (
          <button
            key={p.value}
            type="button"
            aria-pressed={tool.permission === p.value}
            title={p.hint}
            onClick={() => onPermission(p.value)}
            className={`h-6 cursor-pointer rounded-md px-2.5 text-[11px] font-semibold transition-colors ${
              tool.permission === p.value
                ? p.value === "deny"
                  ? "bg-danger-subtle text-danger"
                  : "bg-primary-subtle text-text"
                : "text-text-muted hover:text-text"
            }`}
          >
            {p.label}
          </button>
        ))}
      </div>
      <Button variant="secondary" size="xs" disabled={!tool.available} onClick={onTry}>
        Try
      </Button>
    </div>
  </li>
);

const ConnectorCard: React.FC<{ entry: ConnectorEntry }> = ({ entry }) => {
  const { config, status } = entry;
  const tools = useConnectorsStore((s) => s.tools[config.id]);
  const audit = useConnectorsStore((s) => s.audit[config.id]);
  const busy = useConnectorsStore((s) => s.busy[config.id] === true);
  const connect = useConnectorsStore((s) => s.connect);
  const disconnect = useConnectorsStore((s) => s.disconnect);
  const setEnabled = useConnectorsStore((s) => s.setEnabled);
  const remove = useConnectorsStore((s) => s.remove);
  const setPermission = useConnectorsStore((s) => s.setPermission);
  const loadTools = useConnectorsStore((s) => s.loadTools);
  const loadAudit = useConnectorsStore((s) => s.loadAudit);
  const [open, setOpen] = useState(false);
  const [menu, setMenu] = useState(false);
  const [confirmRemove, setConfirmRemove] = useState(false);
  const [trying, setTrying] = useState<ConnectorTool | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!menu) return;
    const down = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setMenu(false);
    };
    document.addEventListener("mousedown", down);
    return () => document.removeEventListener("mousedown", down);
  }, [menu]);

  useEffect(() => {
    if (open && status.state === "ready") {
      void loadTools(config.id);
      void loadAudit(config.id);
    }
  }, [open, status.state, config.id, loadTools, loadAudit]);

  const where =
    config.kind === "stdio" ? [config.command, ...config.args].join(" ") : (config.url ?? "");

  return (
    <li
      className={`rounded-xl border border-border bg-surface ${config.enabled ? "" : "opacity-70"}`}
    >
      <div className="flex items-center gap-3 p-4">
        <button
          type="button"
          aria-expanded={open}
          aria-label={`${open ? "Hide" : "Show"} ${config.name}`}
          onClick={() => setOpen(!open)}
          className="flex min-w-0 flex-1 cursor-pointer items-center gap-3 text-left"
        >
          <span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-surface-raised text-text-muted">
            <Icon name={config.kind === "stdio" ? "terminal" : "globe"} size={16} />
          </span>
          <span className="min-w-0">
            <span className="block truncate text-sm font-semibold text-text">{config.name}</span>
            <span className="flex items-center gap-2 text-xs text-text-muted">
              <StateDot state={status.state} />
              <span>
                {STATE_TEXT[status.state]}
                {status.state === "ready" &&
                  ` · ${status.toolCount} ${status.toolCount === 1 ? "tool" : "tools"}`}
              </span>
              {status.serverName && (
                <span className="truncate text-text-subtle">{status.serverName}</span>
              )}
            </span>
          </span>
          <Icon
            name="chevron-down"
            size={14}
            className={`ml-auto shrink-0 text-text-muted transition-transform ${open ? "rotate-180" : ""}`}
          />
        </button>
        {status.state === "ready" || status.state === "connecting" ? (
          <Button
            variant="secondary"
            size="sm"
            disabled={busy}
            onClick={() => void disconnect(config.id)}
          >
            Disconnect
          </Button>
        ) : (
          <Button
            variant="primary"
            size="sm"
            disabled={busy || !config.enabled}
            onClick={() => {
              setOpen(true);
              void connect(config.id);
            }}
          >
            Connect
          </Button>
        )}
        <div ref={menuRef} className="relative">
          <button
            type="button"
            aria-label="More actions"
            aria-haspopup="menu"
            aria-expanded={menu}
            onClick={() => setMenu(!menu)}
            className="flex size-8 cursor-pointer items-center justify-center rounded-md text-text-muted transition-colors hover:bg-surface-raised hover:text-text"
          >
            <Icon name="ellipsis" size={16} />
          </button>
          {menu && (
            <div
              role="menu"
              className="absolute right-0 top-9 z-30 w-44 rounded-lg border border-border bg-surface-raised p-1 shadow-lg"
            >
              <button
                type="button"
                role="menuitem"
                onClick={() => {
                  setMenu(false);
                  void setEnabled(config.id, !config.enabled);
                }}
                className="flex w-full cursor-pointer items-center rounded-md px-2.5 py-1.5 text-left text-xs text-text hover:bg-primary-subtle-hover"
              >
                {config.enabled ? "Turn off" : "Turn on"}
              </button>
              <button
                type="button"
                role="menuitem"
                onClick={() => {
                  setMenu(false);
                  setConfirmRemove(true);
                }}
                className="flex w-full cursor-pointer items-center rounded-md px-2.5 py-1.5 text-left text-xs text-danger hover:bg-danger-subtle"
              >
                Remove
              </button>
            </div>
          )}
        </div>
      </div>

      {open && (
        <div className="border-t border-border px-4 pb-4">
          <p className="mt-3 truncate font-mono text-[11px] text-text-subtle" title={where}>
            {where}
          </p>
          {status.state === "error" && status.message && (
            <p
              role="alert"
              className="mt-2 rounded-lg border border-danger/40 bg-danger-subtle px-3 py-2 text-xs text-danger"
            >
              {status.message}
            </p>
          )}
          {status.state === "needs_auth" && (
            <p className="mt-2 rounded-lg border border-border bg-surface-raised px-3 py-2 text-xs text-text-muted">
              This server asks you to sign in first. Signing in comes in the next version.
            </p>
          )}
          {status.state === "ready" && (
            <>
              {tools === undefined ? (
                <Skeleton className="mt-3 h-12 w-full" />
              ) : tools.length === 0 ? (
                <p className="mt-3 text-xs text-text-muted">This connector offers no tools.</p>
              ) : (
                <ul aria-label="Tools" className="mt-1 divide-y divide-border">
                  {tools.map((t) => (
                    <ToolRow
                      key={t.name}
                      tool={t}
                      onPermission={(p) => void setPermission(config.id, t.name, p)}
                      onTry={() => setTrying(t)}
                    />
                  ))}
                </ul>
              )}
              {audit && audit.length > 0 && (
                <details className="mt-3">
                  <summary className="cursor-pointer list-none text-xs font-semibold text-text-muted hover:text-text">
                    Recent calls ({audit.length})
                  </summary>
                  <ul aria-label="Recent calls" className="mt-2 space-y-1">
                    {audit.map((a) => (
                      <li
                        key={a.id}
                        className="flex items-center justify-between gap-3 text-xs text-text-muted"
                      >
                        <span className="truncate">
                          <span className="font-mono text-text">{a.tool}</span> ·{" "}
                          {a.outcome === "needs_approval"
                            ? "waited for approval"
                            : a.outcome === "denied"
                              ? "blocked"
                              : a.outcome}
                        </span>
                        <span className="shrink-0 text-text-subtle">
                          {a.durationMs} ms · {relativeTime(a.createdAt)}
                        </span>
                      </li>
                    ))}
                  </ul>
                </details>
              )}
            </>
          )}
        </div>
      )}

      <ToolTestDialog connectorId={config.id} tool={trying} onClose={() => setTrying(null)} />
      <Dialog
        isOpen={confirmRemove}
        title={`Remove ${config.name}?`}
        description="Its saved permissions, call history and stored variables are deleted from this computer."
        onclose={() => setConfirmRemove(false)}
        actions={
          <>
            <Button variant="ghost" size="sm" onClick={() => setConfirmRemove(false)}>
              Cancel
            </Button>
            <Button
              variant="danger"
              size="sm"
              onClick={() => {
                setConfirmRemove(false);
                void remove(config.id);
              }}
            >
              Remove
            </Button>
          </>
        }
      />
    </li>
  );
};

/** Connectors: the programs and servers that give Dara tools (a calendar, mail). Everything here runs on this computer. */
export const ConnectorsPage: React.FC = () => {
  const entries = useConnectorsStore((s) => s.entries);
  const loading = useConnectorsStore((s) => s.loading);
  const error = useConnectorsStore((s) => s.error);
  const load = useConnectorsStore((s) => s.load);
  const [adding, setAdding] = useState(false);
  const isDemo = getBackend().isDemo;

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="mx-auto w-full max-w-4xl px-8 py-10 text-left">
      <div className="mb-2 flex items-end justify-between gap-4">
        <h1 className="font-[family-name:var(--font-reply)] text-4xl font-semibold tracking-tight text-text">
          Connectors
        </h1>
        <Button variant="primary" size="md" onClick={() => setAdding(true)}>
          <Icon name="plus" size={14} />
          Add connector
        </Button>
      </div>
      <p className="mb-8 max-w-2xl text-sm text-text-muted">
        Connectors give Dara tools such as reading your calendar or searching your mail. Dara asks
        before using a tool unless you allow it. The model runs on this computer; only the tool
        calls reach the connector.
      </p>

      {isDemo && (
        <p className="mb-6 rounded-lg border border-border bg-surface-raised px-4 py-3 text-xs text-text-muted">
          You are looking at a demo with sample data. Connectors run in the Dara desktop app.
        </p>
      )}
      {error && (
        <div
          role="alert"
          className="mb-6 flex items-center justify-between gap-3 rounded-lg border border-danger/40 bg-danger-subtle px-4 py-3 text-xs text-danger"
        >
          <span>{error}</span>
          <Button variant="secondary" size="xs" onClick={() => void load()}>
            Try again
          </Button>
        </div>
      )}

      {loading && entries.length === 0 ? (
        <div className="space-y-3">
          <Skeleton className="h-20 w-full" />
          <Skeleton className="h-20 w-full" />
        </div>
      ) : entries.length === 0 ? (
        !error && (
          <div className="flex flex-col items-center gap-3 rounded-xl border border-dashed border-border px-6 py-14 text-center">
            <Icon name="plug" size={28} className="text-text-subtle" />
            <p className="text-sm font-semibold text-text">No connectors yet</p>
            <p className="max-w-sm text-xs text-text-muted">
              Add a connector to let Dara read a calendar or mail. Nothing is connected until you
              add it, and every tool starts on Ask.
            </p>
            <Button variant="primary" size="sm" onClick={() => setAdding(true)}>
              Add connector
            </Button>
          </div>
        )
      ) : (
        <ul aria-label="Connectors" className="space-y-3">
          {entries.map((e) => (
            <ConnectorCard key={e.config.id} entry={e} />
          ))}
        </ul>
      )}

      <AddConnectorDialog isOpen={adding} onClose={() => setAdding(false)} />
    </div>
  );
};
