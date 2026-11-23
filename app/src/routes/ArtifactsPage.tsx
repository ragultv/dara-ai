import React, { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "../lib/router/useRouter.ts";
import { useArtifactsStore } from "../lib/stores/useArtifactsStore.ts";
import type { ArtifactKind, DbArtifactSummary } from "../lib/api/types.ts";
import { KIND_LABEL, displayTitle, relativeTime } from "../lib/artifacts/format.ts";
import { Markdown } from "../lib/ui/Markdown.tsx";
import { SlideSurface } from "../lib/slides/SlideSurface.tsx";
import { firstSlide } from "../lib/slides/deck.ts";
import { Icon } from "../lib/ui/Icon.tsx";
import { Button } from "../lib/ui/Button.tsx";
import { Dialog } from "../lib/ui/Dialog.tsx";
import { Skeleton } from "../lib/ui/Skeleton.tsx";

const TILES: Array<{ kind: ArtifactKind; icon: string; ready: boolean }> = [
  { kind: "doc", icon: "file-text", ready: true },
  { kind: "slides", icon: "presentation", ready: true },
  { kind: "design", icon: "palette", ready: false }
];

const Card: React.FC<{
  item: DbArtifactSummary;
  onOpen: () => void;
  onDuplicate: () => void;
  onDelete: () => void;
}> = ({ item, onOpen, onDuplicate, onDelete }) => {
  const [menu, setMenu] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!menu) return;
    const down = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setMenu(false);
    };
    document.addEventListener("mousedown", down);
    return () => document.removeEventListener("mousedown", down);
  }, [menu]);

  return (
    <div
      ref={ref}
      className="group relative overflow-hidden rounded-lg border border-border bg-surface transition-colors hover:border-border-strong"
    >
      <button
        type="button"
        onClick={onOpen}
        className="block w-full cursor-pointer text-left focus-visible:outline-2 focus-visible:outline-focus"
        aria-label={`Open ${displayTitle(item.title)}`}
      >
        {item.kind === "slides" ? (
          /* the first slide, as it looks in the deck */
          <div className="relative flex h-44 items-center justify-center overflow-hidden bg-surface-sunken p-4">
            <div className="w-full overflow-hidden rounded-md border border-border shadow-xs">
              <SlideSurface
                className="slide-thumb"
                docId={item.id}
                markdown={firstSlide(item.snippet).slide}
                theme={firstSlide(item.snippet).theme}
              />
            </div>
          </div>
        ) : (
          /* a shrunken copy of the start of the document */
          <div className="relative h-44 overflow-hidden bg-surface-sunken px-5 pt-5">
            <div
              className="pointer-events-none w-[200%] origin-top-left scale-50 select-none"
              aria-hidden="true"
            >
              {item.title && (
                <h3 className="mb-2 font-[family-name:var(--font-reply)] text-[1.6rem] font-semibold leading-tight">
                  {item.title}
                </h3>
              )}
              <Markdown content={item.snippet} />
            </div>
            <div className="pointer-events-none absolute inset-x-0 bottom-0 h-10 bg-linear-to-t from-surface-sunken to-transparent" />
          </div>
        )}
        <div className="border-t border-border px-4 py-3">
          <p className="truncate text-sm font-semibold text-text">{displayTitle(item.title)}</p>
          <p className="mt-0.5 text-xs text-text-muted">
            {KIND_LABEL[item.kind]} <span aria-hidden="true">·</span>{" "}
            {relativeTime(item.updated_at)}
          </p>
        </div>
      </button>

      <button
        type="button"
        aria-label="More actions"
        aria-haspopup="menu"
        aria-expanded={menu}
        onClick={() => setMenu(!menu)}
        className={`absolute right-2 top-2 flex h-7 w-7 cursor-pointer items-center justify-center rounded-md border border-border bg-surface-raised text-text-muted shadow-xs transition-opacity hover:text-text focus-visible:opacity-100 ${
          menu ? "opacity-100" : "opacity-0 group-hover:opacity-100"
        }`}
      >
        <Icon name="ellipsis" size={14} />
      </button>
      {menu && (
        <div
          role="menu"
          className="absolute right-2 top-10 z-30 w-40 rounded-lg border border-border bg-surface-raised p-1 shadow-lg"
        >
          <button
            type="button"
            role="menuitem"
            onClick={() => {
              setMenu(false);
              onDuplicate();
            }}
            className="flex w-full cursor-pointer items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-xs text-text hover:bg-primary-subtle-hover"
          >
            <Icon name="copy" size={13} className="text-text-muted" />
            Duplicate
          </button>
          <button
            type="button"
            role="menuitem"
            onClick={() => {
              setMenu(false);
              onDelete();
            }}
            className="flex w-full cursor-pointer items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-xs text-danger hover:bg-danger-subtle"
          >
            <Icon name="trash-2" size={13} />
            Delete
          </button>
        </div>
      )}
    </div>
  );
};

/** The Artifacts gallery: make something new, and everything made so far. */
export const ArtifactsPage: React.FC = () => {
  const navigate = useRouter((s) => s.navigate);
  const items = useArtifactsStore((s) => s.items);
  const loading = useArtifactsStore((s) => s.loading);
  const error = useArtifactsStore((s) => s.error);
  const load = useArtifactsStore((s) => s.load);
  const create = useArtifactsStore((s) => s.create);
  const duplicate = useArtifactsStore((s) => s.duplicate);
  const remove = useArtifactsStore((s) => s.remove);
  const [query, setQuery] = useState("");
  const [toDelete, setToDelete] = useState<DbArtifactSummary | null>(null);
  const [firstLoad, setFirstLoad] = useState(true);

  useEffect(() => {
    void load().finally(() => setFirstLoad(false));
  }, [load]);

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return items;
    return items.filter(
      (i) => i.title.toLowerCase().includes(q) || i.snippet.toLowerCase().includes(q)
    );
  }, [items, query]);

  const newDoc = async (kind: ArtifactKind = "doc") => {
    const id = await create(kind);
    if (id) navigate("artifact", id);
  };

  return (
    <div className="mx-auto w-full max-w-5xl px-8 py-10 text-left">
      <div className="mb-8 flex items-end justify-between gap-4">
        <h1 className="font-[family-name:var(--font-reply)] text-4xl font-semibold tracking-tight text-text">
          Artifacts
        </h1>
        <div className="flex items-center gap-2">
          <div className="relative">
            <Icon
              name="search"
              size={14}
              className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-text-subtle"
            />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search"
              aria-label="Search artifacts"
              className="h-9 w-56 rounded-lg border border-border bg-surface pl-9 pr-3 text-sm text-text outline-none placeholder:text-text-subtle focus:border-border-strong"
            />
          </div>
          <Button variant="primary" size="md" onClick={() => void newDoc()}>
            <Icon name="plus" size={14} />
            New doc
          </Button>
        </div>
      </div>

      <p className="mb-3 text-xs font-semibold text-text-muted">Make something new</p>
      <div className="mb-10 flex flex-wrap gap-3">
        {TILES.map((t) => (
          <button
            key={t.kind}
            type="button"
            disabled={!t.ready}
            onClick={() => t.ready && void newDoc(t.kind)}
            className="group flex w-36 flex-col gap-2 text-left disabled:cursor-not-allowed"
          >
            <span
              className={`flex h-24 w-36 items-center justify-center rounded-lg border border-border bg-surface-raised text-text-muted transition-colors ${
                t.ready
                  ? "cursor-pointer group-hover:border-border-strong group-hover:text-text"
                  : "opacity-60"
              }`}
            >
              <Icon name={t.icon} size={30} />
            </span>
            <span className="flex items-center gap-2 text-sm font-medium text-text">
              {KIND_LABEL[t.kind]}
              {!t.ready && (
                <span className="rounded-full border border-border px-1.5 py-px text-[10px] font-medium text-text-subtle">
                  Soon
                </span>
              )}
            </span>
          </button>
        ))}
      </div>

      {error && (
        <div
          role="alert"
          className="mb-6 flex items-center justify-between gap-3 rounded-lg border border-danger/40 bg-danger-subtle px-4 py-3 text-xs text-danger"
        >
          <span>Could not load your artifacts: {error}</span>
          <Button variant="secondary" size="xs" onClick={() => void load()}>
            Try again
          </Button>
        </div>
      )}

      {firstLoad && loading ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-60 w-full" />
          ))}
        </div>
      ) : shown.length === 0 ? (
        !error && (
          <div className="flex flex-col items-center gap-3 rounded-lg border border-dashed border-border px-6 py-14 text-center">
            <Icon name="file-text" size={28} className="text-text-subtle" />
            <p className="text-sm font-semibold text-text">
              {query ? "Nothing matches your search" : "No documents yet"}
            </p>
            {!query && (
              <>
                <p className="max-w-sm text-xs text-text-muted">
                  Write a document, or describe one and let Dara write it. Everything stays on this
                  computer.
                </p>
                <Button variant="primary" size="sm" onClick={() => void newDoc()}>
                  New doc
                </Button>
              </>
            )}
          </div>
        )
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {shown.map((item) => (
            <Card
              key={item.id}
              item={item}
              onOpen={() => navigate("artifact", item.id)}
              onDuplicate={() => void duplicate(item.id)}
              onDelete={() => setToDelete(item)}
            />
          ))}
        </div>
      )}

      <Dialog
        isOpen={toDelete !== null}
        title="Delete this document?"
        description={
          toDelete
            ? `"${displayTitle(toDelete.title)}" will be removed from this computer. This cannot be undone.`
            : ""
        }
        onclose={() => setToDelete(null)}
        actions={
          <>
            <Button variant="ghost" size="sm" onClick={() => setToDelete(null)}>
              Cancel
            </Button>
            <Button
              variant="danger"
              size="sm"
              onClick={() => {
                const id = toDelete?.id;
                setToDelete(null);
                if (id) void remove(id);
              }}
            >
              Delete
            </Button>
          </>
        }
      />
    </div>
  );
};
