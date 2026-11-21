import React, { useState } from "react";
import type { ConnectorTool, ToolPermission } from "../../connectors/types.ts";
import { humanizeToolName } from "../../connectors/catalog.ts";
import { useConnectorsStore } from "../../stores/useConnectorsStore.ts";
import { Icon } from "../../ui/Icon.tsx";
import { ToolTestDialog } from "../../connectors/ToolTestDialog.tsx";

const CHOICES: Array<{ value: ToolPermission; label: string; hint: string; icon: string }> = [
  {
    value: "allow",
    label: "Allow",
    hint: "Allow: Dara may use it without asking",
    icon: "circle-check"
  },
  {
    value: "ask",
    label: "Needs approval",
    hint: "Needs approval: Dara asks before each use",
    icon: "hand"
  },
  { value: "deny", label: "Block", hint: "Block: Dara can never use it", icon: "ban" }
];

const ToolRow: React.FC<{
  connectorId: string;
  tool: ConnectorTool;
  onTry: () => void;
}> = ({ connectorId, tool, onTry }) => {
  const setPermission = useConnectorsStore((s) => s.setPermission);
  const [open, setOpen] = useState(false);
  return (
    <li className={`border-t border-border first:border-t-0 ${tool.available ? "" : "opacity-60"}`}>
      <div className="flex items-center gap-3 px-4 py-2.5">
        <button
          type="button"
          aria-expanded={open}
          aria-label={`${open ? "Hide" : "Show"} details of ${tool.title ?? humanizeToolName(tool.name)}`}
          onClick={() => setOpen(!open)}
          className="flex min-w-0 flex-1 cursor-pointer items-center gap-4 text-left"
        >
          <span className="w-52 shrink-0 truncate text-sm text-text">
            {tool.title ?? humanizeToolName(tool.name)}
          </span>
          <span className="min-w-0 flex-1 truncate text-xs text-text-muted">
            {tool.description}
          </span>
        </button>
        {tool.changed && (
          <span
            className="shrink-0 rounded-full border border-warning/40 bg-warning-subtle px-1.5 py-px text-[10px] font-medium text-warning"
            title="This tool changed since you last chose a permission, so it was put back to Needs approval."
          >
            changed
          </span>
        )}
        <div
          role="group"
          aria-label={`Permission for ${tool.name}`}
          className="flex shrink-0 gap-1"
        >
          {CHOICES.map((c) => (
            <button
              key={c.value}
              type="button"
              aria-pressed={tool.permission === c.value}
              aria-label={`${c.label} ${tool.name}`}
              title={c.hint}
              onClick={() => void setPermission(connectorId, tool.name, c.value)}
              className={`flex size-8 cursor-pointer items-center justify-center rounded-lg border transition-colors ${
                tool.permission === c.value
                  ? c.value === "deny"
                    ? "border-danger/50 bg-danger-subtle text-danger"
                    : "border-border-strong bg-surface-raised text-text"
                  : "border-transparent text-text-subtle hover:bg-surface-raised hover:text-text"
              }`}
            >
              <Icon name={c.icon} size={16} />
            </button>
          ))}
        </div>
        <button
          type="button"
          aria-label={`${open ? "Hide" : "Show"} details`}
          onClick={() => setOpen(!open)}
          className="shrink-0 cursor-pointer text-text-subtle"
        >
          <Icon name="chevron-down" size={14} className={open ? "rotate-180" : ""} />
        </button>
      </div>
      {open && (
        <div className="space-y-2 px-4 pb-3 text-xs text-text-muted">
          <p className="whitespace-pre-wrap leading-relaxed">{tool.description}</p>
          <p className="font-mono text-[11px] text-text-subtle">{tool.name}</p>
          {tool.readOnlyHint === true && (
            <p>The server says this tool only reads. Dara does not rely on that: you decide.</p>
          )}
          {!tool.available && <p>The server no longer offers this tool.</p>}
          <button
            type="button"
            disabled={!tool.available}
            onClick={onTry}
            className="cursor-pointer rounded-md border border-border px-2.5 py-1 text-xs font-semibold text-text transition-colors hover:bg-surface-raised disabled:cursor-not-allowed disabled:opacity-50"
          >
            Try this tool
          </button>
        </div>
      )}
    </li>
  );
};

const Group: React.FC<{
  title: string;
  note: string;
  connectorId: string;
  tools: ConnectorTool[];
  onTry: (t: ConnectorTool) => void;
}> = ({ title, note, connectorId, tools, onTry }) => {
  const setPermission = useConnectorsStore((s) => s.setPermission);
  if (tools.length === 0) return null;
  const same = tools.every((t) => t.permission === tools[0]!.permission);
  const current = same ? tools[0]!.permission : null;
  const setAll = async (p: ToolPermission) => {
    for (const t of tools) if (t.permission !== p) await setPermission(connectorId, t.name, p);
  };
  return (
    <section className="mt-7" aria-label={title}>
      <div className="mb-2 flex items-end justify-between gap-4">
        <div>
          <h3 className="text-sm font-semibold text-text">{title}</h3>
          <p className="text-xs text-text-muted">{note}</p>
        </div>
        <label className="flex shrink-0 items-center gap-2 text-xs text-text-muted">
          <span className="sr-only">Set every tool in {title}</span>
          <select
            aria-label={`Set every tool in ${title}`}
            value={current ?? "mixed"}
            onChange={(e) =>
              e.target.value !== "mixed" && void setAll(e.target.value as ToolPermission)
            }
            className="h-8 cursor-pointer rounded-lg border border-border bg-surface-raised px-2 text-xs font-semibold text-text outline-none"
          >
            {current === null && <option value="mixed">Mixed</option>}
            <option value="allow">Allow all</option>
            <option value="ask">Needs approval for all</option>
            <option value="deny">Block all</option>
          </select>
        </label>
      </div>
      <ul className="rounded-xl border border-border bg-surface">
        {tools.map((t) => (
          <ToolRow key={t.name} connectorId={connectorId} tool={t} onTry={() => onTry(t)} />
        ))}
      </ul>
    </section>
  );
};

/** What Dara may do with each tool of one connector: allow, ask first, or block. */
export const PermissionsTab: React.FC<{ connectorId: string; connected: boolean }> = ({
  connectorId,
  connected
}) => {
  const tools = useConnectorsStore((s) => s.tools[connectorId]);
  const [trying, setTrying] = useState<ConnectorTool | null>(null);

  if (!connected || tools === undefined)
    return (
      <p className="mt-6 text-sm text-text-muted">
        Connect to see this connector's tools and choose what Dara may do with each one.
      </p>
    );
  if (tools.length === 0)
    return <p className="mt-6 text-sm text-text-muted">This connector offers no tools.</p>;

  const reads = tools.filter((t) => t.readOnlyHint === true);
  const others = tools.filter((t) => t.readOnlyHint !== true);
  return (
    <div className="text-left">
      <p className="mt-4 max-w-2xl text-sm text-text-muted">
        Choose when Dara may use each tool. Every tool starts on Needs approval; Dara asks before
        each use. A tool that changes after you chose is put back to Needs approval.
      </p>
      <Group
        title="Tools the server says only read"
        note="The server's own claim. Dara does not rely on it."
        connectorId={connectorId}
        tools={reads}
        onTry={setTrying}
      />
      <Group
        title="Other tools"
        note="These can change things, or the server does not say they only read."
        connectorId={connectorId}
        tools={others}
        onTry={setTrying}
      />
      <ToolTestDialog connectorId={connectorId} tool={trying} onClose={() => setTrying(null)} />
    </div>
  );
};
