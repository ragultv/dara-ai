import React, { useEffect, useRef, useState } from "react";
import { EditorContent } from "@tiptap/react";
import { useRouter } from "../lib/router/useRouter.ts";
import { useArtifactsStore } from "../lib/stores/useArtifactsStore.ts";
import { useModelsStore } from "../lib/stores/useModelsStore.ts";
import { useDocEditor } from "../lib/artifacts/useDocEditor.ts";
import { DocToolbar } from "../lib/artifacts/DocToolbar.tsx";
import { TableControls } from "../lib/artifacts/TableControls.tsx";
import { SelectionMenu } from "../lib/artifacts/SelectionMenu.tsx";
import { DocChat } from "../lib/artifacts/DocChat.tsx";
import { useDocAssistant } from "../lib/artifacts/useDocAssistant.ts";
import { displayTitle, exportFileName, relativeTime } from "../lib/artifacts/format.ts";
import { Icon } from "../lib/ui/Icon.tsx";
import { Button } from "../lib/ui/Button.tsx";
import { Dialog } from "../lib/ui/Dialog.tsx";
import { Skeleton } from "../lib/ui/Skeleton.tsx";

const SAVE_TEXT = { idle: "", saving: "Saving...", saved: "Saved", error: "Not saved" } as const;

/** One open document: title, editor, the formatting bar, and the Ask Dara prompt. */
const DocView: React.FC = () => {
  const doc = useArtifactsStore((s) => s.doc)!;
  const saveState = useArtifactsStore((s) => s.saveState);
  const setTitle = useArtifactsStore((s) => s.setTitle);
  const setContent = useArtifactsStore((s) => s.setContent);
  const remove = useArtifactsStore((s) => s.remove);
  const duplicate = useArtifactsStore((s) => s.duplicate);
  const navigate = useRouter((s) => s.navigate);
  const modelLoaded = useModelsStore((s) => s.models.some((m) => m.dara.loaded));

  const editor = useDocEditor({ initial: doc.content, onChange: setContent });
  const titleRef = useRef<HTMLInputElement>(null);
  const [pageEl, setPageEl] = useState<HTMLDivElement | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [chatOpen, setChatOpen] = useState(true);
  const { busy, send, stop, undo } = useDocAssistant(editor, doc.id);

  useEffect(() => {
    if (!menuOpen) return;
    const down = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setMenuOpen(false);
    };
    document.addEventListener("mousedown", down);
    return () => document.removeEventListener("mousedown", down);
  }, [menuOpen]);

  const exportMarkdown = () => {
    const body = `${doc.title ? `# ${doc.title}\n\n` : ""}${doc.content}`;
    const url = URL.createObjectURL(new Blob([body], { type: "text/markdown" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = exportFileName(doc.title);
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  return (
    <div className="flex h-full bg-bg text-text">
      <div className="relative flex min-w-0 flex-1 flex-col">
        {/* top bar: back, title menu, save state */}
        <div className="flex h-12 shrink-0 items-center justify-between border-b border-border px-4">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => navigate("artifacts")}
              aria-label="Back to artifacts"
              title="Back to artifacts"
              className="flex h-8 w-8 cursor-pointer items-center justify-center rounded-md text-text-muted transition-colors hover:bg-surface-raised hover:text-text"
            >
              <Icon name="arrow-left" size={16} />
            </button>
            <div ref={menuRef} className="relative">
              <button
                type="button"
                aria-haspopup="menu"
                aria-expanded={menuOpen}
                onClick={() => setMenuOpen(!menuOpen)}
                className="flex h-8 max-w-[320px] cursor-pointer items-center gap-1.5 rounded-md px-2.5 text-sm font-semibold transition-colors hover:bg-surface-raised"
              >
                <span className="truncate">{displayTitle(doc.title)}</span>
                <Icon name="chevron-down" size={13} className="shrink-0 text-text-muted" />
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
          <div className="flex items-center gap-2">
            <p
              role="status"
              aria-live="polite"
              className={`text-xs ${saveState === "error" ? "text-danger" : "text-text-muted"}`}
            >
              {SAVE_TEXT[saveState]}
            </p>
            {!chatOpen && (
              <button
                type="button"
                onClick={() => setChatOpen(true)}
                className="flex h-8 cursor-pointer items-center gap-1.5 rounded-md px-2.5 text-xs font-semibold text-text-muted transition-colors hover:bg-surface-raised hover:text-text"
              >
                <Icon name="message-square" size={14} />
                Show chat
              </button>
            )}
          </div>
        </div>

        {/* the page */}
        <div className="min-h-0 flex-1 overflow-y-auto custom-scrollbar">
          <div className="mx-auto w-full max-w-3xl px-8 pb-40 pt-12">
            <input
              ref={titleRef}
              value={doc.title}
              onChange={(e) => setTitle(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  editor?.commands.focus("start");
                }
              }}
              placeholder="Title"
              aria-label="Document title"
              maxLength={500}
              className="doc-title mb-4 w-full border-0 bg-transparent p-0 font-[family-name:var(--font-reply)] text-4xl font-semibold leading-tight tracking-tight text-text outline-none placeholder:text-text-subtle"
            />
            {editor ? (
              <div ref={setPageEl} className="relative">
                <EditorContent editor={editor} />
                {/* the selection menu is moved out of this wrapper by its tooltip: keep it before anything React adds or removes */}
                <SelectionMenu editor={editor} />
                <TableControls editor={editor} container={pageEl} disabled={busy} />
              </div>
            ) : (
              <Skeleton className="h-40 w-full" />
            )}
          </div>
        </div>

        {/* floating formatting bar */}
        {editor && (
          <div className="pointer-events-none absolute inset-x-0 bottom-4 flex justify-center px-6">
            <div className="pointer-events-auto">
              <DocToolbar editor={editor} disabled={busy} />
            </div>
          </div>
        )}
      </div>

      {chatOpen && (
        <DocChat
          docId={doc.id}
          docIsEmpty={!doc.content.trim()}
          busy={busy}
          modelLoaded={modelLoaded}
          onSend={(t) => void send(t)}
          onStop={stop}
          onUndo={undo}
          onClose={() => setChatOpen(false)}
          onOpenModels={() => navigate("models")}
        />
      )}

      <Dialog
        isOpen={confirmDelete}
        title="Delete this document?"
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

export const ArtifactEditorPage: React.FC = () => {
  const id = useRouter((s) => s.artifactId);
  const navigate = useRouter((s) => s.navigate);
  const doc = useArtifactsStore((s) => s.doc);
  const docLoading = useArtifactsStore((s) => s.docLoading);
  const docError = useArtifactsStore((s) => s.docError);
  const open = useArtifactsStore((s) => s.open);

  useEffect(() => {
    if (id) void open(id);
    return () => {
      void useArtifactsStore.getState().close();
    };
  }, [id, open]);

  if (docLoading || (!doc && !docError)) {
    return (
      <div className="mx-auto w-full max-w-3xl space-y-4 px-8 pt-16">
        <Skeleton className="h-10 w-1/2" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-5/6" />
      </div>
    );
  }
  if (!doc) {
    return (
      <div className="mx-auto flex h-full max-w-md flex-col items-center justify-center gap-3 px-6 text-center">
        <p className="text-sm font-semibold text-text">This document could not be opened</p>
        <p className="text-xs text-text-muted">{docError}</p>
        <Button variant="secondary" size="sm" onClick={() => navigate("artifacts")}>
          Back to artifacts
        </Button>
      </div>
    );
  }
  return <DocView key={doc.id} />;
};
