import React, { useMemo, useState } from "react";
import { useConnectorsStore } from "../../stores/useConnectorsStore.ts";
import { Icon } from "../../ui/Icon.tsx";
import { Skeleton } from "../../ui/Skeleton.tsx";
import { toast } from "../../ui/Toast.tsx";
import { ConnectorAvatar } from "./ConnectorAvatar.tsx";
import { buildItems, statusText, type ConnectorItem } from "./items.ts";
import { addFromCatalog } from "./addFromCatalog.ts";

type Filter = "all" | "added";

const Card: React.FC<{ item: ConnectorItem; onOpen: () => void; onAdd: () => void }> = ({
  item,
  onOpen,
  onAdd
}) => {
  const added = item.connector !== undefined;
  return (
    <li className="relative">
      <button
        type="button"
        onClick={onOpen}
        aria-label={`${item.name}${added ? ", added" : ""}`}
        className="flex w-full cursor-pointer gap-3.5 rounded-xl px-3.5 py-3.5 text-left transition-colors hover:bg-surface-raised focus-visible:outline-2 focus-visible:outline-focus"
      >
        <ConnectorAvatar name={item.name} />
        <span className="min-w-0 flex-1 pr-9">
          <span className="flex items-center gap-2 text-sm font-semibold text-text">
            {item.name}
            {item.soon && (
              <span className="rounded-full border border-border px-1.5 py-px text-[10px] font-medium text-text-subtle">
                Soon
              </span>
            )}
          </span>
          <span className="mt-0.5 line-clamp-2 block text-xs leading-relaxed text-text-muted">
            {item.tagline}
          </span>
          <span className="mt-1 block text-[11px] text-text-subtle">
            {added ? statusText(item.connector) : <>Connector · By {item.publisher}</>}
          </span>
        </span>
      </button>
      {!item.soon &&
        (added ? (
          <span
            aria-hidden="true"
            className="pointer-events-none absolute right-4 top-4 flex size-7 items-center justify-center rounded-lg text-text-muted"
          >
            <Icon name="check" size={15} />
          </span>
        ) : (
          <button
            type="button"
            aria-label={`Add ${item.name}`}
            title={`Add ${item.name}`}
            onClick={onAdd}
            className="absolute right-4 top-4 flex size-7 cursor-pointer items-center justify-center rounded-lg bg-surface-raised text-text-muted transition-colors hover:bg-primary-subtle-hover hover:text-text"
          >
            <Icon name="plus" size={15} />
          </button>
        ))}
    </li>
  );
};

/** The apps view: what you have added, and what Dara suggests, as cards. */
export const CatalogList: React.FC<{
  onOpen: (key: string) => void;
  onAddCustom: () => void;
}> = ({ onOpen, onAddCustom }) => {
  const entries = useConnectorsStore((s) => s.entries);
  const loading = useConnectorsStore((s) => s.loading);
  const error = useConnectorsStore((s) => s.error);
  const load = useConnectorsStore((s) => s.load);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<Filter>("all");
  const [menu, setMenu] = useState(false);

  const items = useMemo(() => buildItems(entries), [entries]);
  const q = query.trim().toLowerCase();
  const shown = items.filter(
    (i) =>
      (filter === "all" || i.connector) &&
      (!q || `${i.name} ${i.tagline} ${i.publisher}`.toLowerCase().includes(q))
  );
  const added = shown.filter((i) => i.connector);
  const available = shown.filter((i) => !i.connector);

  const add = async (item: ConnectorItem) => {
    if (!item.catalog) return;
    // a server with no sign-in needs its API key first: its page asks for it
    if (item.catalog.registers === "api-key") return onOpen(item.key);
    const problem = await addFromCatalog(item.catalog);
    if (problem) toast.show(problem, "error");
    else onOpen(item.key);
  };

  return (
    <div className="text-left">
      <h1 className="font-[family-name:var(--font-reply)] text-3xl font-semibold tracking-tight text-text">
        Connectors
      </h1>
      <p className="mt-1 max-w-2xl text-sm text-text-muted">
        Connect Dara to your mail, calendar and tools. The model runs on this computer; only the
        tool calls reach the connector, and Dara asks before using any tool unless you allow it.
      </p>

      <div className="mt-6 flex flex-wrap items-center gap-2">
        <div className="relative min-w-56 flex-1">
          <Icon
            name="search"
            size={14}
            className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-text-subtle"
          />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search connectors"
            aria-label="Search connectors"
            className="h-9 w-full rounded-lg border border-border bg-surface pl-9 pr-3 text-sm text-text outline-none placeholder:text-text-subtle focus:border-border-strong"
          />
        </div>
        <div role="group" aria-label="Show" className="flex rounded-lg border border-border p-0.5">
          {(["all", "added"] as const).map((f) => (
            <button
              key={f}
              type="button"
              aria-pressed={filter === f}
              onClick={() => setFilter(f)}
              className={`h-7 cursor-pointer rounded-md px-3 text-xs font-semibold transition-colors ${
                filter === f ? "bg-surface-raised text-text" : "text-text-muted hover:text-text"
              }`}
            >
              {f === "all" ? "All" : "Added"}
            </button>
          ))}
        </div>
        <div className="relative">
          <button
            type="button"
            aria-haspopup="menu"
            aria-expanded={menu}
            onClick={() => setMenu(!menu)}
            className="flex h-9 cursor-pointer items-center gap-1.5 rounded-lg bg-primary px-4 text-xs font-semibold text-white transition-colors hover:bg-primary-hover"
          >
            Add
            <Icon name="chevron-down" size={13} />
          </button>
          {menu && (
            <div
              role="menu"
              className="absolute right-0 top-11 z-30 w-60 rounded-lg border border-border bg-surface-raised p-1 shadow-lg"
              onMouseLeave={() => setMenu(false)}
            >
              <button
                type="button"
                role="menuitem"
                onClick={() => {
                  setMenu(false);
                  onAddCustom();
                }}
                className="flex w-full cursor-pointer flex-col rounded-md px-2.5 py-2 text-left hover:bg-primary-subtle-hover"
              >
                <span className="text-xs font-semibold text-text">Add custom connector</span>
                <span className="text-[11px] text-text-muted">
                  A program on this computer, or an MCP server address
                </span>
              </button>
            </div>
          )}
        </div>
      </div>

      {error && (
        <div
          role="alert"
          className="mt-4 flex items-center justify-between gap-3 rounded-lg border border-danger/40 bg-danger-subtle px-4 py-3 text-xs text-danger"
        >
          <span>{error}</span>
          <button
            type="button"
            className="cursor-pointer font-semibold underline"
            onClick={() => void load()}
          >
            Try again
          </button>
        </div>
      )}

      {loading && entries.length === 0 ? (
        <div className="mt-6 grid gap-3 sm:grid-cols-2">
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-24 w-full" />
        </div>
      ) : shown.length === 0 ? (
        <p className="mt-10 text-center text-sm text-text-muted">
          {filter === "added" && !q
            ? "You have not added a connector yet."
            : "No connector matches."}
        </p>
      ) : (
        <>
          {added.length > 0 && (
            <section className="mt-8" aria-label="Added connectors">
              <h2 className="mb-1 text-sm font-semibold text-text">Added</h2>
              <ul className="grid gap-x-6 sm:grid-cols-2">
                {added.map((i) => (
                  <Card
                    key={i.key}
                    item={i}
                    onOpen={() => onOpen(i.key)}
                    onAdd={() => void add(i)}
                  />
                ))}
              </ul>
            </section>
          )}
          {available.length > 0 && filter === "all" && (
            <section className="mt-8" aria-label="Available connectors">
              <h2 className="mb-1 text-sm font-semibold text-text">Available</h2>
              <ul className="grid gap-x-6 sm:grid-cols-2">
                {available.map((i) => (
                  <Card
                    key={i.key}
                    item={i}
                    onOpen={() => onOpen(i.key)}
                    onAdd={() => void add(i)}
                  />
                ))}
              </ul>
            </section>
          )}
        </>
      )}
    </div>
  );
};
