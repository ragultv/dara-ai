import React, { useEffect, useMemo, useRef, useState } from "react";
import { EditorContent } from "@tiptap/react";
import { useRouter } from "../lib/router/useRouter.ts";
import { useArtifactsStore } from "../lib/stores/useArtifactsStore.ts";
import { useModelsStore } from "../lib/stores/useModelsStore.ts";
import { useCommentsStore } from "../lib/stores/useCommentsStore.ts";
import { setMarkdown, useDocEditor } from "../lib/artifacts/useDocEditor.ts";
import { DeckView } from "../lib/slides/DeckView.tsx";
import { DocToolbar } from "../lib/artifacts/DocToolbar.tsx";
import { TableControls } from "../lib/artifacts/TableControls.tsx";
import { SelectionMenu } from "../lib/artifacts/SelectionMenu.tsx";
import { DocChat, type PanelTab } from "../lib/artifacts/DocChat.tsx";
import { DocComments } from "../lib/artifacts/DocComments.tsx";
import { CommentComposer, ThreadPopover } from "../lib/artifacts/CommentPopovers.tsx";
import {
  findQuote,
  quoteForSelection,
  refreshCommentMarks
} from "../lib/artifacts/extensions/commentMarks.ts";
import { insertMedia } from "../lib/artifacts/extensions/media.ts";
import {
  buildThreads,
  commentInstruction,
  mentionsDara,
  type Thread
} from "../lib/artifacts/threads.ts";
import { editorTarget, useDocAssistant } from "../lib/artifacts/useDocAssistant.ts";
import { displayTitle, exportFileName, relativeTime } from "../lib/artifacts/format.ts";
import { Icon } from "../lib/ui/Icon.tsx";
import { Button } from "../lib/ui/Button.tsx";
import { Dialog } from "../lib/ui/Dialog.tsx";
import { Skeleton } from "../lib/ui/Skeleton.tsx";
import { toast } from "../lib/ui/Toast.tsx";
import { storeApi } from "../lib/api/store-api.ts";
import { buildDocx, readExportImage, type PMNode } from "../lib/artifacts/export/docx.ts";
import { downloadBlob } from "../lib/artifacts/export/download.ts";
import { printDocument } from "../lib/artifacts/export/print.ts";
import type { DbArtifactComment } from "../lib/api/types.ts";

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

  const [chatOpen, setChatOpen] = useState(true);
  const [panelTab, setPanelTab] = useState<PanelTab>("chat");
  const [draft, setDraft] = useState<{ quote: string; nth: number } | null>(null);
  const draftRef = useRef<{ quote: string; nth: number } | null>(null);
  const [threadOpen, setThreadOpen] = useState(false);
  const [prefill, setPrefill] = useState<{ text: string; nonce: number } | null>(null);
  const showPanel = (tab: PanelTab) => {
    setChatOpen(true);
    setPanelTab(tab);
  };

  const editor = useDocEditor({
    initial: doc.content,
    onChange: (md) => setContent(md),
    docId: doc.id,
    comments: {
      getAnchors: () => [
        ...useCommentsStore
          .getState()
          .items.filter((c) => c.parent_id === null && !c.resolved && c.quote)
          .map((c) => ({ id: c.id, quote: c.quote, nth: c.nth })),
        ...(draftRef.current
          ? [{ id: "draft", quote: draftRef.current.quote, nth: draftRef.current.nth }]
          : [])
      ],
      getActive: () => useCommentsStore.getState().activeId,
      onSelect: (id) => {
        if (id === "draft") return;
        useCommentsStore.getState().setActive(id);
        setThreadOpen(true);
      }
    },
    onMediaProblem: (m) => toast.show(m, "error")
  });
  const titleRef = useRef<HTMLInputElement>(null);
  const [pageEl, setPageEl] = useState<HTMLDivElement | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const target = useMemo(() => (editor ? editorTarget(editor) : null), [editor]);
  const { busy, send, stop, undo } = useDocAssistant(target, doc.id);

  // the comments belong to the document: read them when it opens, redraw the highlights when they change
  const comments = useCommentsStore((s) => s.items);
  const activeThread = useCommentsStore((s) => s.activeId);
  useEffect(() => {
    void useCommentsStore.getState().load(doc.id);
    return () => useCommentsStore.getState().reset();
  }, [doc.id]);
  useEffect(() => {
    draftRef.current = draft;
    if (editor) refreshCommentMarks(editor);
  }, [editor, comments, activeThread, draft]);
  const openThreads = useMemo(
    () => buildThreads(comments).filter((t) => !t.root.resolved).length,
    [comments]
  );

  const openThread = useMemo(
    () => buildThreads(comments).find((t) => t.root.id === activeThread) ?? null,
    [comments, activeThread]
  );

  const startComment = () => {
    if (!editor) return;
    const q = quoteForSelection(editor);
    if (!q) {
      toast.show("Select some words, or put the cursor in a paragraph, to comment on it.", "info");
      return;
    }
    setThreadOpen(false);
    setDraft({ quote: q.quote, nth: q.nth });
  };

  /** Wakes Dara for a thread: she reads it, changes the document if it calls for that, and answers in the thread. */
  const askAboutThread = (rootId: string, thread: Thread, display: string) =>
    void send(commentInstruction(thread), {
      display,
      onResult: async (r) => {
        if (r.stopped) return;
        await useCommentsStore
          .getState()
          .add({ author: "dara", parent_id: rootId, body: r.summary });
      }
    });

  /** A comment that could not be saved, so Dara can still be asked about it (her answer then goes to the chat only). */
  const unsaved = (
    body: string,
    extra: { quote?: string; nth?: number; parent_id?: string | null }
  ): DbArtifactComment => ({
    id: "unsaved",
    parent_id: extra.parent_id ?? null,
    author: "user",
    body,
    quote: extra.quote ?? "",
    nth: extra.nth ?? 0,
    resolved: false,
    created_at: Date.now(),
    updated_at: Date.now()
  });
  const saveProblem = () => {
    const why = useCommentsStore.getState().error ?? "The comment could not be saved.";
    toast.show(`${why} Dara can still answer; the comment will not be kept.`, "error");
  };

  const submitDraft = async (body: string, askDara: boolean) => {
    if (!draft) return;
    const c = await useCommentsStore.getState().add({ body, quote: draft.quote, nth: draft.nth });
    setDraft(null);
    if (!c) {
      saveProblem();
      if (askDara)
        void send(commentInstruction({ root: unsaved(body, draft), replies: [] }), {
          display: `Comment: ${body.slice(0, 100)}`
        });
      return;
    }
    useCommentsStore.getState().setActive(c.id);
    if (askDara) askAboutThread(c.id, { root: c, replies: [] }, `Comment: ${body.slice(0, 100)}`);
  };

  const replyToThread = async (thread: Thread, body: string) => {
    const c = await useCommentsStore.getState().add({ body, parent_id: thread.root.id });
    const daraTookPart =
      thread.root.author === "dara" || thread.replies.some((r) => r.author === "dara");
    const asks = daraTookPart || mentionsDara(body);
    if (!c) {
      saveProblem();
      if (asks)
        void send(
          commentInstruction({
            root: thread.root,
            replies: [...thread.replies, unsaved(body, { parent_id: thread.root.id })]
          }),
          { display: `Reply: ${body.slice(0, 100)}` }
        );
      return;
    }
    if (asks)
      askAboutThread(
        thread.root.id,
        { root: thread.root, replies: [...thread.replies, c] },
        `Reply: ${body.slice(0, 100)}`
      );
  };

  const focusThread = (root: { id: string; quote: string; nth: number }) => {
    if (!editor) return;
    const hit = findQuote(editor.state.doc, root.quote, root.nth);
    if (!hit) {
      toast.show("That text is no longer in the document.", "info");
      return;
    }
    editor.chain().focus().setTextSelection(hit).scrollIntoView().run();
    useCommentsStore.getState().setActive(root.id);
    setThreadOpen(true);
  };

  const askDara = () => {
    if (!editor) return;
    const { from, to } = editor.state.selection;
    const picked =
      from !== to ? editor.state.doc.textBetween(from, to, " ").trim().slice(0, 300) : "";
    setPrefill({ text: picked ? `About "${picked}": ` : "", nonce: Date.now() });
    showPanel("chat");
  };

  const addMedia = (file: File) => {
    if (editor)
      void insertMedia(editor, doc.id, file).then(
        (problem) => problem && toast.show(problem, "error")
      );
  };

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
    downloadBlob(new Blob([body], { type: "text/markdown" }), exportFileName(doc.title));
  };

  /** Word: a real .docx with the page's structure and look; images are included when Word can show them (PNG, JPEG, GIF). */
  const exportWord = async () => {
    if (!editor) return;
    try {
      const blob = await buildDocx(editor.getJSON() as PMNode, {
        title: doc.title,
        loadImage: async (assetId) => {
          try {
            return await readExportImage(await storeApi.getArtifactAsset(doc.id, assetId));
          } catch {
            return null;
          }
        }
      });
      downloadBlob(blob, exportFileName(doc.title, "docx"));
    } catch {
      toast.show("The Word file could not be created.", "error");
    }
  };

  /** PDF: the page is printed in its own look; choose "Save as PDF" in the print window. */
  const exportPdf = () => {
    if (!editor) return;
    toast.show('In the print window, choose "Save as PDF".', "info");
    void printDocument(editor, doc.title);
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
                    { label: "Export as PDF", icon: "file-text", run: exportPdf },
                    { label: "Export as Word", icon: "file", run: exportWord },
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
                Show panel
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
                <SelectionMenu editor={editor} onComment={startComment} />
                <TableControls editor={editor} container={pageEl} disabled={busy} />
                {draft && (
                  <CommentComposer
                    editor={editor}
                    container={pageEl}
                    draft={draft}
                    modelLoaded={modelLoaded}
                    busy={busy}
                    onSubmit={(b, ask) => void submitDraft(b, ask)}
                    onCancel={() => setDraft(null)}
                  />
                )}
                {!draft && threadOpen && openThread && (
                  <ThreadPopover
                    editor={editor}
                    container={pageEl}
                    thread={openThread}
                    busy={busy}
                    onReply={(b) => void replyToThread(openThread, b)}
                    onClose={() => {
                      setThreadOpen(false);
                      useCommentsStore.getState().setActive(null);
                    }}
                  />
                )}
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
              <DocToolbar
                editor={editor}
                disabled={busy}
                onMedia={addMedia}
                onComment={startComment}
                onAsk={askDara}
              />
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
          tab={panelTab}
          onTab={setPanelTab}
          openThreads={openThreads}
          prefill={prefill}
          commentsSlot={
            <DocComments
              busy={busy}
              onReply={(t, b) => void replyToThread(t, b)}
              onFocusThread={focusThread}
            />
          }
          updatedAt={doc.updated_at}
          onRestored={(md) => editor && setMarkdown(editor, md)}
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
  return doc.kind === "slides" ? <DeckView key={doc.id} /> : <DocView key={doc.id} />;
};
