import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import type { Editor } from "@tiptap/react";
import type { DbArtifactComment } from "../api/types.ts";
import { useCommentsStore } from "../stores/useCommentsStore.ts";
import { Icon } from "../ui/Icon.tsx";
import { CommentMessage, ReplyBox } from "./CommentParts.tsx";
import { findQuote } from "./extensions/commentMarks.ts";
import type { Thread } from "./threads.ts";

// Comments the way Claude shows them: a small card next to the words. Writing a new comment opens one beside the selection, and a click on
// highlighted words opens that thread beside them. Both are positioned inside the page wrapper, from the position of the words.

const CARD_WIDTH = 320;

/** Where to put a card: just under the first line of the words, kept inside the wrapper. Follows edits, resizes and scrolling content. */
function useAnchorBox(
  editor: Editor,
  container: HTMLElement | null,
  anchor: { quote: string; nth: number } | null
): { left: number; top: number } | null {
  const [box, setBox] = useState<{ left: number; top: number } | null>(null);
  const quote = anchor?.quote ?? null;
  const nth = anchor?.nth ?? 0;
  const put = (next: { left: number; top: number } | null) =>
    setBox((prev) =>
      prev && next && prev.left === next.left && prev.top === next.top ? prev : next
    );
  const measure = useCallback(() => {
    if (!container || quote === null) return put(null);
    const hit = findQuote(editor.state.doc, quote, nth);
    if (!hit) return put(null);
    try {
      const c = editor.view.coordsAtPos(hit.from);
      const base = container.getBoundingClientRect();
      const room = Math.max(0, base.width - CARD_WIDTH);
      put({
        left: Math.round(Math.min(Math.max(0, c.left - base.left), room)),
        top: Math.round(c.bottom - base.top + 8)
      });
    } catch {
      put(null);
    }
  }, [editor, container, quote, nth]);

  useLayoutEffect(() => {
    measure();
  }, [measure]);
  useEffect(() => {
    const on = () => requestAnimationFrame(measure);
    editor.on("transaction", on);
    window.addEventListener("resize", on);
    const ro = container ? new ResizeObserver(on) : null;
    if (container && ro) ro.observe(container);
    return () => {
      editor.off("transaction", on);
      window.removeEventListener("resize", on);
      ro?.disconnect();
    };
  }, [editor, container, measure]);
  return box;
}

const card = "absolute z-30 rounded-xl border border-border bg-surface-raised p-3 shadow-lg";

/** The card for a new comment: the words it is about, a box, "Send to Dara" and the send arrow. */
export const CommentComposer: React.FC<{
  editor: Editor;
  container: HTMLElement | null;
  draft: { quote: string; nth: number };
  modelLoaded: boolean;
  busy: boolean;
  onSubmit: (body: string, askDara: boolean) => void;
  onCancel: () => void;
}> = ({ editor, container, draft, modelLoaded, busy, onSubmit, onCancel }) => {
  const box = useAnchorBox(editor, container, draft);
  const [body, setBody] = useState("");
  const canAsk = modelLoaded && !busy;
  const [askDara, setAskDara] = useState(true);
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    ref.current?.focus();
  }, []);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 140)}px`;
  }, [body]);

  const send = () => {
    const v = body.trim();
    if (v) onSubmit(v, askDara && canAsk);
  };

  if (!box) return null;
  return (
    <div
      role="dialog"
      aria-label="New comment"
      style={{ left: box.left, top: box.top, width: CARD_WIDTH }}
      className={card}
      onKeyDown={(e) => e.key === "Escape" && onCancel()}
    >
      <p className="mb-2 inline-flex items-center rounded-md bg-primary-subtle px-1.5 py-0.5 text-xs font-medium text-primary">
        @Dara
      </p>
      <textarea
        ref={ref}
        value={body}
        rows={2}
        onChange={(e) => setBody(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            send();
          }
        }}
        placeholder="Add a comment"
        aria-label="New comment text"
        className="custom-scrollbar block w-full resize-none border-0 bg-transparent text-[13px] text-text outline-none placeholder:text-text-subtle"
      />
      <div className="mt-2 flex items-center justify-between">
        <label
          className={`flex items-center gap-2 text-xs ${canAsk ? "cursor-pointer text-text" : "text-text-subtle"}`}
        >
          <input
            type="checkbox"
            checked={askDara && canAsk}
            disabled={!canAsk}
            onChange={(e) => setAskDara(e.target.checked)}
            className="size-3.5 accent-[var(--color-primary)]"
          />
          Send to Dara
        </label>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={onCancel}
            aria-label="Cancel comment"
            title="Cancel"
            className="flex size-7 cursor-pointer items-center justify-center rounded-md text-text-muted transition-colors hover:bg-surface-sunken hover:text-text"
          >
            <Icon name="x" size={14} />
          </button>
          <button
            type="button"
            onClick={send}
            disabled={!body.trim()}
            aria-label="Send comment"
            title="Send"
            className="flex size-7 cursor-pointer items-center justify-center rounded-md bg-primary text-white transition-colors hover:bg-primary-hover disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Icon name="arrow-up" size={14} />
          </button>
        </div>
      </div>
    </div>
  );
};

/** The card for an existing thread, opened by a click on its highlighted words. */
export const ThreadPopover: React.FC<{
  editor: Editor;
  container: HTMLElement | null;
  thread: Thread;
  busy: boolean;
  onReply: (body: string) => void;
  onClose: () => void;
}> = ({ editor, container, thread, busy, onReply, onClose }) => {
  const { root, replies } = thread;
  const box = useAnchorBox(editor, container, { quote: root.quote, nth: root.nth });
  const setResolved = useCommentsStore((s) => s.setResolved);
  if (!box) return null;
  const all: DbArtifactComment[] = [root, ...replies];
  return (
    <div
      role="dialog"
      aria-label="Comment thread"
      style={{ left: box.left, top: box.top, width: CARD_WIDTH }}
      className={card}
      onKeyDown={(e) => e.key === "Escape" && onClose()}
    >
      <div className="mb-2 flex items-center justify-end gap-0.5">
        <button
          type="button"
          onClick={() => {
            void setResolved(root.id, true);
            onClose();
          }}
          aria-label="Resolve thread"
          title="Resolve"
          className="flex size-7 cursor-pointer items-center justify-center rounded-md text-text-muted transition-colors hover:bg-surface-sunken hover:text-text"
        >
          <Icon name="check" size={14} />
        </button>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close thread"
          title="Close"
          className="flex size-7 cursor-pointer items-center justify-center rounded-md text-text-muted transition-colors hover:bg-surface-sunken hover:text-text"
        >
          <Icon name="x" size={14} />
        </button>
      </div>
      <div className="max-h-72 space-y-3 overflow-y-auto pr-1 custom-scrollbar">
        {all.map((c) => (
          <CommentMessage key={c.id} c={c} />
        ))}
      </div>
      <div className="mt-3">
        <ReplyBox
          placeholder={root.author === "dara" ? "Reply to Dara" : "Reply, or write @dara"}
          disabled={busy}
          onSubmit={onReply}
        />
      </div>
    </div>
  );
};
