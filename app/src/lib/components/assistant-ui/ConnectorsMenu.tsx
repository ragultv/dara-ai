import React, { useEffect, useRef, useState } from "react";
import { Icon } from "../../ui/Icon.tsx";
import { useConnectorsStore } from "../../stores/useConnectorsStore.ts";
import { useChatToolsStore } from "../../stores/useChatToolsStore.ts";
import { logoForName } from "../../connectors/catalog.ts";
import type { ConnectorEntry, ConnectorTool } from "../../connectors/types.ts";

const SETUP_HREF = "#/settings?tab=connectors";
const NONE: string[] = [];

/** A compact version of ConnectorAvatar (logo, or first letter) for menu rows. */
export const SmallAvatar: React.FC<{ name: string }> = ({ name }) => {
  const logo = logoForName(name);
  return (
    <span
      aria-hidden="true"
      className="flex size-7 shrink-0 items-center justify-center rounded-lg border border-border bg-white font-[family-name:var(--font-reply)] text-xs font-semibold text-text"
    >
      {logo ? (
        <img src={logo} alt="" className="size-4" draggable={false} />
      ) : (
        name.trim().charAt(0).toUpperCase() || "?"
      )}
    </span>
  );
};

/** One line about whether a connector can be used in chat right now. */
function describe(
  entry: ConnectorEntry,
  tools: ConnectorTool[] | undefined
): { line: string; usable: boolean } {
  const { config, status } = entry;
  if (!config.enabled) return { line: "Turned off", usable: false };
  switch (status.state) {
    case "ready": {
      const n = tools
        ? tools.filter((t) => t.available && t.permission !== "deny").length
        : status.toolCount;
      return { line: `Ready · ${n} ${n === 1 ? "tool" : "tools"}`, usable: true };
    }
    case "needs_auth":
      return { line: "Not signed in", usable: false };
    case "error":
      return { line: "Could not connect", usable: false };
    case "connecting":
      return { line: "Connecting…", usable: true };
    default:
      return { line: "Connects when used", usable: true };
  }
}

/** The plug button in the composer: choose which connectors the model may use in this chat. */
export const ConnectorsMenu: React.FC<{ conversationId: string | null; disabled?: boolean }> = ({
  conversationId,
  disabled = false
}) => {
  const entries = useConnectorsStore((s) => s.entries);
  const tools = useConnectorsStore((s) => s.tools);
  const load = useConnectorsStore((s) => s.load);
  const enabled = useChatToolsStore((s) => s.enabled[conversationId ?? "__new__"] ?? NONE);
  const toggle = useChatToolsStore((s) => s.toggle);
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);

  // Only count connectors that still exist.
  const activeCount = enabled.filter((id) => entries.some((e) => e.config.id === id)).length;
  const count = entries.length ? activeCount : enabled.length;

  useEffect(() => {
    if (!open) return;
    if (entries.length === 0) void load();
    const onDown = (e: MouseEvent) => {
      if (root.current && !root.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const on = count > 0;

  return (
    <div ref={root} className="relative">
      <button
        type="button"
        disabled={disabled}
        onClick={() => setOpen((v) => !v)}
        aria-label={on ? `Connectors: ${count} on in this chat` : "Connectors"}
        aria-haspopup="dialog"
        aria-expanded={open}
        title="Connectors"
        className={`h-6 px-2 rounded-full border transition-all flex items-center gap-1.5 text-xs font-medium select-none cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed ${
          on
            ? "bg-text/15 border-border-strong text-text shadow-xs hover:bg-text/25"
            : "bg-surface-sunken border-border text-text-subtle hover:text-text hover:bg-surface-raised"
        }`}
      >
        <Icon name="plug" size={12} />
        {on && <span className="text-[11px] font-semibold tabular-nums">{count}</span>}
      </button>

      {open && (
        <div
          role="dialog"
          aria-label="Connectors in this chat"
          className="absolute bottom-full left-0 z-40 mb-2 w-80 max-w-[calc(100vw-2rem)] rounded-lg border border-border bg-surface-raised p-2 shadow-lg"
        >
          <div className="flex items-baseline justify-between px-2 pt-1 pb-2">
            <p className="text-[10px] font-bold uppercase tracking-wider text-text-subtle">
              Connectors
            </p>
            <a
              href={SETUP_HREF}
              onClick={() => setOpen(false)}
              className="text-[11px] font-medium text-text-muted hover:text-text"
            >
              Manage
            </a>
          </div>

          {entries.length === 0 ? (
            <div className="px-2 py-4 text-center">
              <p className="text-xs font-medium text-text">No connectors yet</p>
              <a
                href={SETUP_HREF}
                onClick={() => setOpen(false)}
                className="mt-1 inline-block text-xs font-semibold text-primary hover:underline"
              >
                Add connectors
              </a>
            </div>
          ) : (
            <ul className="max-h-72 overflow-y-auto custom-scrollbar">
              {entries.map((entry) => {
                const id = entry.config.id;
                const { line, usable } = describe(entry, tools[id]);
                const checked = enabled.includes(id);
                const canSwitch = usable || checked;
                return (
                  <li
                    key={id}
                    className="flex items-center gap-2.5 rounded-md px-2 py-1.5 hover:bg-surface-sunken"
                  >
                    <SmallAvatar name={entry.config.name} />
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-xs font-semibold text-text">
                        {entry.config.name}
                      </p>
                      <p
                        className={`truncate text-[11px] ${usable ? "text-text-muted" : "text-text-subtle"}`}
                      >
                        {line}
                        {!usable && (
                          <>
                            {" · "}
                            <a
                              href={SETUP_HREF}
                              onClick={() => setOpen(false)}
                              className="font-medium text-primary hover:underline"
                            >
                              Set up
                            </a>
                          </>
                        )}
                      </p>
                    </div>
                    <button
                      type="button"
                      role="switch"
                      aria-checked={checked}
                      aria-label={`Use ${entry.config.name} in this chat`}
                      disabled={!canSwitch}
                      onClick={() => toggle(conversationId, id)}
                      className={`relative h-4.5 w-8 shrink-0 rounded-full transition-colors cursor-pointer disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline-2 focus-visible:outline-focus focus-visible:outline-offset-2 ${
                        checked ? "bg-primary" : "bg-border-strong"
                      }`}
                    >
                      <span
                        className={`absolute top-0.5 left-0.5 size-3.5 rounded-full bg-white shadow-sm transition-transform ${
                          checked ? "translate-x-3.5" : ""
                        }`}
                      />
                    </button>
                  </li>
                );
              })}
            </ul>
          )}

          <p className="mt-1 border-t border-border px-2 pt-2 pb-1 text-[11px] leading-snug text-text-muted">
            Dara asks before using a tool unless you allowed it.
          </p>
        </div>
      )}
    </div>
  );
};
