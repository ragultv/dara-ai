import React, { useEffect, useMemo, useRef, useState } from "react";
import { EditorContent } from "@tiptap/react";
import { useRouter } from "../router/useRouter.ts";
import { useArtifactsStore } from "../stores/useArtifactsStore.ts";
import { useModelsStore } from "../stores/useModelsStore.ts";
import { DocChat, type PanelTab } from "../artifacts/DocChat.tsx";
import { DocToolbar } from "../artifacts/DocToolbar.tsx";
import { insertMedia } from "../artifacts/extensions/media.ts";
import { useDocAssistant, type DocTarget } from "../artifacts/useDocAssistant.ts";
import { displayTitle, exportFileName, relativeTime } from "../artifacts/format.ts";
import { downloadBlob } from "../artifacts/export/download.ts";
import { Icon } from "../ui/Icon.tsx";
import { Button } from "../ui/Button.tsx";
import { Dialog } from "../ui/Dialog.tsx";
import { toast } from "../ui/Toast.tsx";
import {
  THEMES,
  addSlide,
  clampIndex,
  deckIsEmpty,
  deleteSlide,
  duplicateSlide,
  isBlank,
  isTheme,
  moveSlide,
  parseDeck,
  replaceSlide,
  serializeDeck,
  slideTitle,
  type Deck
} from "./deck.ts";
import { Present } from "./Present.tsx";
import { SlideSurface } from "./SlideSurface.tsx";
import { printDeck } from "./printDeck.ts";
import { useSlideEditor } from "./useSlideEditor.ts";

const SAVE_TEXT = { idle: "", saving: "Saving...", saved: "Saved", error: "Not saved" } as const;

/** The page of the slide being edited: the slide itself is the editor. One of these per slide, so each keeps its own undo history. */
const SlideEditor: React.FC<{
  docId: string;
  theme: Deck["theme"];
  initial: string;
  onChange: (markdown: string) => void;
  onAsk: () => void;
}> = ({ docId, theme, initial, onChange, onAsk }) => {
  const [live, setLive] = useState(initial);
  const editor = useSlideEditor({
    initial,
    docId,
    onChange: (md) => {
      setLive(md);
      onChange(md);
    },
    onMediaProblem: (m) => toast.show(m, "error")
  });
  return (
    <>
      <div className="w-full overflow-hidden rounded-lg border border-border shadow-sm">
        <SlideSurface markdown={live} theme={theme} docId={docId}>
          {editor ? <EditorContent editor={editor} /> : null}
        </SlideSurface>
      </div>
      {editor && (
        <div className="pointer-events-none absolute inset-x-0 bottom-4 flex justify-center px-4">
          <div className="pointer-events-auto max-w-full">
            <DocToolbar
              editor={editor}
              wrap
              onMedia={(file) =>
                void insertMedia(editor, docId, file).then(
                  (problem) => problem && toast.show(problem, "error")
                )
              }
              onAsk={onAsk}
            />
          </div>
        </div>
      )}
    </>
  );
};

const IconButton: React.FC<{
  label: string;
  icon: string;
  disabled?: boolean;
  onClick: () => void;
  danger?: boolean;
}> = ({ label, icon, disabled, onClick, danger }) => (
  <button
    type="button"
    aria-label={label}
    title={label}
    disabled={disabled}
    onClick={(e) => {
      e.stopPropagation();
      onClick();
    }}
    className={`flex size-6 cursor-pointer items-center justify-center rounded-md bg-surface/90 text-text-muted shadow-xs transition-colors hover:bg-surface-raised disabled:cursor-default disabled:opacity-30 ${
      danger ? "hover:text-danger" : "hover:text-text"
    }`}
  >
    <Icon name={icon} size={13} />
  </button>
);

/** The left column: every slide as a small page, in order. Drag one to move it; the buttons on a slide add, copy, move and remove. */
const SlideList: React.FC<{
  deck: Deck;
  docId: string;
  index: number;
  disabled: boolean;
  onSelect: (i: number) => void;
  onAdd: () => void;
  onDuplicate: (i: number) => void;
  onDelete: (i: number) => void;
  onMove: (from: number, to: number) => void;
}> = ({ deck, docId, index, disabled, onSelect, onAdd, onDuplicate, onDelete, onMove }) => {
  const dragFrom = useRef<number | null>(null);
  const [over, setOver] = useState<number | null>(null);
  const selected = useRef<HTMLLIElement>(null);
  useEffect(() => {
    selected.current?.scrollIntoView({ block: "nearest" });
  }, [index]);

  return (
    <nav
      aria-label="Slides"
      className="flex h-full w-56 shrink-0 flex-col border-r border-border bg-surface-sunken"
    >
      <ol className="custom-scrollbar min-h-0 flex-1 space-y-3 overflow-y-auto px-3 py-3">
        {deck.slides.map((s, i) => (
          <li
            key={i}
            ref={i === index ? selected : undefined}
            draggable={!disabled}
            onDragStart={(e) => {
              dragFrom.current = i;
              e.dataTransfer.effectAllowed = "move";
            }}
            onDragOver={(e) => {
              if (dragFrom.current === null) return;
              e.preventDefault();
              setOver(i);
            }}
            onDragLeave={() => setOver((o) => (o === i ? null : o))}
            onDrop={(e) => {
              e.preventDefault();
              const from = dragFrom.current;
              dragFrom.current = null;
              setOver(null);
              if (from !== null) onMove(from, i);
            }}
            onDragEnd={() => {
              dragFrom.current = null;
              setOver(null);
            }}
            className="group relative flex gap-2"
          >
            <span
              className={`w-4 shrink-0 pt-1 text-right text-[11px] tabular-nums ${
                i === index ? "font-semibold text-text" : "text-text-subtle"
              }`}
            >
              {i + 1}
            </span>
            <button
              type="button"
              onClick={() => onSelect(i)}
              aria-label={`Slide ${i + 1}${slideTitle(s) ? `: ${slideTitle(s)}` : " (blank)"}`}
              aria-current={i === index}
              className={`relative block min-w-0 flex-1 cursor-pointer overflow-hidden rounded-md border-2 text-left transition-colors focus-visible:outline-2 focus-visible:outline-focus ${
                i === index
                  ? "border-primary"
                  : over === i
                    ? "border-primary/60"
                    : "border-border hover:border-border-strong"
              }`}
            >
              <SlideSurface markdown={s} theme={deck.theme} docId={docId} className="slide-thumb" />
              {isBlank(s) && (
                <span className="pointer-events-none absolute inset-0 flex items-center justify-center text-[10px] text-text-subtle">
                  Blank slide
                </span>
              )}
            </button>
            <div className="absolute right-1 top-1 flex flex-col gap-0.5 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100">
              <IconButton
                label="Move up"
                icon="arrow-up"
                disabled={disabled || i === 0}
                onClick={() => onMove(i, i - 1)}
              />
              <IconButton
                label="Move down"
                icon="arrow-down"
                disabled={disabled || i === deck.slides.length - 1}
                onClick={() => onMove(i, i + 1)}
              />
              <IconButton
                label="Duplicate slide"
                icon="copy"
                disabled={disabled}
                onClick={() => onDuplicate(i)}
              />
              <IconButton
                label="Delete slide"
                icon="trash-2"
                danger
                disabled={disabled}
                onClick={() => onDelete(i)}
              />
            </div>
          </li>
        ))}
      </ol>
      <div className="shrink-0 border-t border-border p-3">
        <Button
          variant="secondary"
          size="sm"
          className="w-full"
          disabled={disabled}
          onClick={onAdd}
        >
          <Icon name="plus" size={14} />
          New slide
        </Button>
      </div>
    </nav>
  );
};

/** One open deck: slide list, the slide being edited, the theme, Present, and the Dara panel. */
export const DeckView: React.FC = () => {
  const doc = useArtifactsStore((s) => s.doc)!;
  const saveState = useArtifactsStore((s) => s.saveState);
  const setTitle = useArtifactsStore((s) => s.setTitle);
  const setContent = useArtifactsStore((s) => s.setContent);
  const remove = useArtifactsStore((s) => s.remove);
  const duplicate = useArtifactsStore((s) => s.duplicate);
  const navigate = useRouter((s) => s.navigate);
  const modelLoaded = useModelsStore((s) => s.models.some((m) => m.dara.loaded));

  const [index, setIndex] = useState(0);
  const [present, setPresent] = useState(false);
  const [chatOpen, setChatOpen] = useState(true);
  const [panelTab, setPanelTab] = useState<PanelTab>("chat");
  const [prefill, setPrefill] = useState<{ text: string; nonce: number } | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const titleRef = useRef<HTMLInputElement>(null);

  // What the page shows while Dara writes: her live text. Otherwise the saved deck.
  const [preview, setPreview] = useState<string | null>(null);
  const stored = useMemo(() => parseDeck(doc.content), [doc.content]);
  const shown = useMemo(() => (preview !== null ? parseDeck(preview) : stored), [preview, stored]);
  const idx = clampIndex(shown, index);
  const idxRef = useRef(idx);
  idxRef.current = idx;

  // Text this page wrote itself. Any other change (Dara, Undo, a restored version) starts a new editor so it shows the new text.
  const wrote = useRef(doc.content);
  const [rev, setRev] = useState(0);
  useEffect(() => {
    if (doc.content !== wrote.current) {
      wrote.current = doc.content;
      setRev((r) => r + 1);
    }
  }, [doc.content]);

  const commit = (deck: Deck, snapshot?: string) => {
    const md = serializeDeck(deck);
    wrote.current = md;
    setContent(md, snapshot ? { snapshot } : undefined);
  };
  /** A change to the list of slides: the editor is started again, because the slide under it may be another one now. */
  const restructure = (deck: Deck, to: number, snapshot?: string) => {
    commit(deck, snapshot);
    setRev((r) => r + 1);
    setIndex(clampIndex(deck, to));
  };

  const target = useMemo<DocTarget>(
    () => ({
      get: () => useArtifactsStore.getState().doc?.content ?? "",
      set: (md) => setPreview(md),
      setEditable: () => {}
    }),
    []
  );
  const { busy, send, stop, undo } = useDocAssistant(target, doc.id, "slides");
  useEffect(() => {
    if (!busy) setPreview(null);
  }, [busy, doc.content]);

  useEffect(() => {
    if (!menuOpen) return;
    const down = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setMenuOpen(false);
    };
    document.addEventListener("mousedown", down);
    return () => document.removeEventListener("mousedown", down);
  }, [menuOpen]);

  const askDara = () => {
    setPrefill({ text: "", nonce: Date.now() });
    setChatOpen(true);
    setPanelTab("chat");
  };

  const current = () => parseDeck(useArtifactsStore.getState().doc?.content ?? "");
  const exportMarkdown = () =>
    downloadBlob(new Blob([doc.content], { type: "text/markdown" }), exportFileName(doc.title));
  const exportPdf = () => {
    toast.show('In the print window, choose "Save as PDF".', "info");
    void printDeck(current(), doc.title, doc.id);
  };

  if (present)
    return (
      <Present
        deck={stored}
        docId={doc.id}
        title={doc.title}
        start={idx}
        onExit={() => setPresent(false)}
      />
    );

  return (
    <div className="flex h-full bg-bg text-text">
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex h-12 shrink-0 items-center justify-between gap-3 border-b border-border px-4">
          <div className="flex min-w-0 items-center gap-2">
            <button
              type="button"
              onClick={() => navigate("artifacts")}
              aria-label="Back to artifacts"
              title="Back to artifacts"
              className="flex h-8 w-8 shrink-0 cursor-pointer items-center justify-center rounded-md text-text-muted transition-colors hover:bg-surface-raised hover:text-text"
            >
              <Icon name="arrow-left" size={16} />
            </button>
            <input
              ref={titleRef}
              value={doc.title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Untitled deck"
              aria-label="Deck title"
              maxLength={500}
              className="h-8 min-w-0 max-w-[320px] flex-1 rounded-md border-0 bg-transparent px-2 text-sm font-semibold text-text outline-none transition-colors placeholder:text-text-subtle hover:bg-surface-raised focus:bg-surface-raised"
            />
            <div ref={menuRef} className="relative">
              <button
                type="button"
                aria-label="Deck menu"
                aria-haspopup="menu"
                aria-expanded={menuOpen}
                onClick={() => setMenuOpen(!menuOpen)}
                className="flex h-8 w-8 cursor-pointer items-center justify-center rounded-md text-text-muted transition-colors hover:bg-surface-raised hover:text-text"
              >
                <Icon name="chevron-down" size={14} />
              </button>
              {menuOpen && (
                <div
                  role="menu"
                  className="absolute left-0 top-full z-40 mt-1.5 w-56 rounded-lg border border-border bg-surface-raised p-1.5 shadow-lg"
                >
                  <p className="px-2.5 pb-1 pt-1 text-[11px] text-text-subtle">
                    Updated {relativeTime(doc.updated_at)}
                  </p>
                  {[
                    { label: "Rename", icon: "pencil", run: () => titleRef.current?.focus() },
                    {
                      label: "Duplicate",
                      icon: "copy",
                      run: async () => {
                        const id = await duplicate(doc.id);
                        if (id) navigate("artifact", id);
                      }
                    },
                    { label: "Export as PDF", icon: "file-text", run: exportPdf },
                    { label: "Export as Markdown", icon: "download", run: exportMarkdown }
                  ].map((item) => (
                    <button
                      key={item.label}
                      type="button"
                      role="menuitem"
                      onClick={() => {
                        setMenuOpen(false);
                        void item.run();
                      }}
                      className="flex w-full cursor-pointer items-center gap-2.5 rounded-md px-2.5 py-2 text-left text-xs text-text transition-colors hover:bg-primary-subtle-hover"
                    >
                      <Icon name={item.icon} size={14} className="text-text-muted" />
                      {item.label}
                    </button>
                  ))}
                  <div className="my-1 border-t border-border" />
                  <button
                    type="button"
                    role="menuitem"
                    onClick={() => {
                      setMenuOpen(false);
                      setConfirmDelete(true);
                    }}
                    className="flex w-full cursor-pointer items-center gap-2.5 rounded-md px-2.5 py-2 text-left text-xs text-danger transition-colors hover:bg-danger-subtle"
                  >
                    <Icon name="trash-2" size={14} />
                    Delete
                  </button>
                </div>
              )}
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <p
              role="status"
              aria-live="polite"
              className={`text-xs ${saveState === "error" ? "text-danger" : "text-text-muted"}`}
            >
              {SAVE_TEXT[saveState]}
            </p>
            <label className="flex items-center gap-1.5 text-xs text-text-muted">
              <Icon name="palette" size={14} />
              <span className="sr-only">Theme</span>
              <select
                value={shown.theme}
                disabled={busy}
                onChange={(e) => {
                  const t = e.target.value;
                  if (isTheme(t)) commit({ ...current(), theme: t }, undefined);
                }}
                className="h-8 cursor-pointer rounded-md border border-border bg-surface px-2 text-xs text-text outline-none focus-visible:outline-2 focus-visible:outline-focus"
              >
                {THEMES.map((t) => (
                  <option key={t.id} value={t.id}>
                    {t.label}
                  </option>
                ))}
              </select>
            </label>
            <Button
              variant="primary"
              size="sm"
              disabled={busy || deckIsEmpty(doc.content)}
              onClick={() => setPresent(true)}
            >
              <Icon name="presentation" size={14} />
              Present
            </Button>
            {!chatOpen && (
              <button
                type="button"
                onClick={() => setChatOpen(true)}
                className="flex h-8 cursor-pointer items-center gap-1.5 rounded-md px-2.5 text-xs font-semibold text-text-muted transition-colors hover:bg-surface-raised hover:text-text"
              >
                <Icon name="message-square" size={14} />
                Show panel
              </button>
            )}
          </div>
        </div>

        <div className="flex min-h-0 flex-1">
          <SlideList
            deck={shown}
            docId={doc.id}
            index={idx}
            disabled={busy}
            onSelect={setIndex}
            onAdd={() => restructure(addSlide(current(), idx), idx + 1)}
            onDuplicate={(i) => restructure(duplicateSlide(current(), i), i + 1)}
            onDelete={(i) =>
              restructure(
                deleteSlide(current(), i),
                Math.min(i, shown.slides.length - 2),
                `Before deleting slide ${i + 1}`
              )
            }
            onMove={(from, to) => {
              if (from !== to && to >= 0 && to < shown.slides.length)
                restructure(moveSlide(current(), from, to), to);
            }}
          />
          <div className="relative flex min-w-0 flex-1 items-start justify-center overflow-y-auto px-8 pb-24 pt-8 custom-scrollbar">
            <div
              className="w-full"
              style={{ maxWidth: "min(100%, calc((100vh - 15rem) * 16 / 9))" }}
            >
              {busy ? (
                <div className="w-full overflow-hidden rounded-lg border border-border shadow-sm">
                  <SlideSurface
                    markdown={shown.slides[idx] ?? ""}
                    theme={shown.theme}
                    docId={doc.id}
                  />
                </div>
              ) : (
                <SlideEditor
                  key={`${idx}:${rev}`}
                  docId={doc.id}
                  theme={shown.theme}
                  initial={shown.slides[idx] ?? ""}
                  onAsk={askDara}
                  onChange={(md) => commit(replaceSlide(current(), idxRef.current, md))}
                />
              )}
              <p className="mt-3 text-center text-xs text-text-subtle">
                Slide {idx + 1} of {shown.slides.length}
                {busy ? " · Dara is working" : ""}
              </p>
            </div>
          </div>
        </div>
      </div>

      {chatOpen && (
        <DocChat
          kind="slides"
          docId={doc.id}
          docIsEmpty={deckIsEmpty(doc.content)}
          busy={busy}
          modelLoaded={modelLoaded}
          onSend={(t) => void send(t)}
          onStop={stop}
          onUndo={undo}
          onClose={() => setChatOpen(false)}
          onOpenModels={() => navigate("models")}
          tab={panelTab}
          onTab={setPanelTab}
          prefill={prefill}
          updatedAt={doc.updated_at}
          onRestored={() => {}}
        />
      )}

      <Dialog
        isOpen={confirmDelete}
        title="Delete this deck?"
        description={`"${displayTitle(doc.title)}" will be removed from this computer. This cannot be undone.`}
        onclose={() => setConfirmDelete(false)}
        actions={
          <>
            <Button variant="ghost" size="sm" onClick={() => setConfirmDelete(false)}>
              Cancel
            </Button>
            <Button
              variant="danger"
              size="sm"
              onClick={() => {
                setConfirmDelete(false);
                void remove(doc.id).then((ok) => ok && navigate("artifacts"));
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
