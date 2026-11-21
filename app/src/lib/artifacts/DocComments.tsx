import React, { useEffect, useMemo, useRef } from "react";
import type { DbArtifactComment } from "../api/types.ts";
import { useCommentsStore } from "../stores/useCommentsStore.ts";
import { Icon } from "../ui/Icon.tsx";
import { CommentMessage, ReplyBox } from "./CommentParts.tsx";
import { awaitsDara, buildThreads, type Thread } from "./threads.ts";

interface Props {
  /** Dara is working on something, so asking her again has to wait */
  busy: boolean;
  onReply: (thread: Thread, body: string) => void;
  /** scroll the page to the words a thread points at */
  onFocusThread: (root: DbArtifactComment) => void;
}

const ThreadCard: React.FC<{
  thread: Thread;
  active: boolean;
  busy: boolean;
  onActivate: () => void;
  onReply: (body: string) => void;
  onFocus: () => void;
}> = ({ thread, active, busy, onActivate, onReply, onFocus }) => {
  const setResolved = useCommentsStore((s) => s.setResolved);
  const remove = useCommentsStore((s) => s.remove);
  const ref = useRef<HTMLLIElement>(null);
  const { root, replies } = thread;
  const waiting = awaitsDara(thread);

  useEffect(() => {
    if (active) ref.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [active]);

  return (
    <li
      ref={ref}
      data-thread={root.id}
      onClick={onActivate}
      className={`rounded-xl border p-3 transition-colors ${
        active ? "border-border-strong bg-surface-raised" : "border-border bg-surface"
      } ${root.resolved ? "opacity-70" : ""}`}
    >
      {root.quote && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onFocus();
          }}
          title="Show in the document"
          className="mb-3 line-clamp-2 w-full cursor-pointer rounded-md bg-warning-subtle px-2 py-1 text-left text-xs text-text hover:brightness-95"
        >
          {root.quote}
        </button>
      )}
      <div className="space-y-3">
        {[root, ...replies].map((c) => (
          <CommentMessage key={c.id} c={c} />
        ))}
      </div>
      {!root.resolved && (
        <div className="mt-3" onClick={(e) => e.stopPropagation()}>
          <ReplyBox
            placeholder={root.author === "dara" ? "Reply to Dara" : "Reply, or write @dara"}
            disabled={busy}
            onSubmit={onReply}
          />
          {waiting && busy && (
            <p className="reasoning-shimmer mt-1.5 text-[11px] text-text-muted">
              Dara is working...
            </p>
          )}
        </div>
      )}
      <div
        className="mt-2 flex items-center justify-end gap-1"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          type="button"
          onClick={() => void setResolved(root.id, !root.resolved)}
          className="flex h-6 cursor-pointer items-center gap-1 rounded-md px-1.5 text-[11px] text-text-muted transition-colors hover:bg-surface-sunken hover:text-text"
        >
          <Icon name="check" size={12} />
          {root.resolved ? "Reopen" : "Resolve"}
        </button>
        <button
          type="button"
          onClick={() => void remove(root.id)}
          aria-label="Delete thread"
          title="Delete thread"
          className="flex size-6 cursor-pointer items-center justify-center rounded-md text-text-muted transition-colors hover:bg-danger-subtle hover:text-danger"
        >
          <Icon name="trash-2" size={12} />
        </button>
      </div>
    </li>
  );
};

/** The Comments tab: every thread in the document, open ones first. A new comment is written in a card beside the words, not here. */
export const DocComments: React.FC<Props> = ({ busy, onReply, onFocusThread }) => {
  const items = useCommentsStore((s) => s.items);
  const error = useCommentsStore((s) => s.error);
  const activeId = useCommentsStore((s) => s.activeId);
  const setActive = useCommentsStore((s) => s.setActive);
  const threads = useMemo(() => buildThreads(items), [items]);
  const open = threads.filter((t) => !t.root.resolved);
  const resolved = threads.filter((t) => t.root.resolved);

  const renderCard = (t: Thread) => (
    <ThreadCard
      key={t.root.id}
      thread={t}
      active={activeId === t.root.id}
      busy={busy}
      onActivate={() => setActive(t.root.id)}
      onReply={(b) => onReply(t, b)}
      onFocus={() => onFocusThread(t.root)}
    />
  );

  return (
    <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3 custom-scrollbar">
      {error && (
        <p
          role="alert"
          className="rounded-lg border border-border bg-surface-raised px-3 py-2 text-xs text-text-muted"
        >
          {error}
        </p>
      )}

      {open.length === 0 && !error && (
        <p className="px-1 pt-2 text-xs leading-relaxed text-text-muted">
          No comments yet. Select some text and choose the comment button to start a thread. If Dara
          is unsure about something she asks here instead of guessing, and replying to her makes her
          act on your answer.
        </p>
      )}

      <ul aria-label="Comment threads" className="space-y-3">
        {open.map(renderCard)}
      </ul>

      {resolved.length > 0 && (
        <details>
          <summary className="cursor-pointer list-none px-1 text-xs font-semibold text-text-muted hover:text-text">
            Resolved ({resolved.length})
          </summary>
          <ul className="mt-2 space-y-3">{resolved.map(renderCard)}</ul>
        </details>
      )}
    </div>
  );
};
