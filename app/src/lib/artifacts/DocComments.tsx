import React, { useEffect, useMemo, useRef, useState } from "react";
import type { DbArtifactComment } from "../api/types.ts";
import { useCommentsStore } from "../stores/useCommentsStore.ts";
import { Icon } from "../ui/Icon.tsx";
import { Button } from "../ui/Button.tsx";
import { relativeTime } from "./format.ts";
import { awaitsDara, buildThreads, type Thread } from "./threads.ts";

interface Props {
  /** a new thread being written: the words it points at */
  draft: { quote: string; nth: number } | null;
  /** Dara is working on something, so asking her again has to wait */
  busy: boolean;
  modelLoaded: boolean;
  onDraftSubmit: (body: string, askDara: boolean) => void;
  onDraftCancel: () => void;
  onReply: (thread: Thread, body: string) => void;
  /** scroll the page to the words a thread points at */
  onFocusThread: (root: DbArtifactComment) => void;
}

const Author: React.FC<{ c: DbArtifactComment }> = ({ c }) => (
  <div className="flex items-center gap-1.5 text-[11px] text-text-subtle">
    {c.author === "dara" ? (
      <Icon name="sparkles" size={11} className="text-text-muted" />
    ) : (
      <Icon name="user" size={11} className="text-text-muted" />
    )}
    <span className="font-semibold text-text-muted">{c.author === "dara" ? "Dara" : "You"}</span>
    <span aria-hidden="true">·</span>
    <span>{relativeTime(c.created_at)}</span>
  </div>
);

const ReplyBox: React.FC<{
  placeholder: string;
  disabled?: boolean;
  onSubmit: (body: string) => void;
}> = ({ placeholder, disabled, onSubmit }) => {
  const [text, setText] = useState("");
  const submit = () => {
    const v = text.trim();
    if (!v || disabled) return;
    setText("");
    onSubmit(v);
  };
  return (
    <div className="flex items-end gap-1.5">
      <textarea
        value={text}
        rows={1}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
        placeholder={placeholder}
        aria-label={placeholder}
        className="custom-scrollbar max-h-24 min-h-8 flex-1 resize-none rounded-md border border-border bg-bg px-2.5 py-1.5 text-xs text-text outline-none placeholder:text-text-subtle focus:border-border-strong"
      />
      <button
        type="button"
        onClick={submit}
        disabled={!text.trim() || disabled}
        aria-label="Reply"
        title="Reply"
        className="flex h-8 w-8 shrink-0 cursor-pointer items-center justify-center rounded-md bg-primary text-white transition-colors hover:bg-primary-hover disabled:cursor-not-allowed disabled:opacity-40"
      >
        <Icon name="send" size={13} />
      </button>
    </div>
  );
};

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
      className={`rounded-lg border p-3 transition-colors ${
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
          className="mb-2 line-clamp-2 w-full cursor-pointer border-l-2 border-primary/60 pl-2 text-left text-xs italic text-text-muted hover:text-text"
        >
          {root.quote}
        </button>
      )}
      <div className="space-y-2.5">
        {[root, ...replies].map((c) => (
          <div key={c.id}>
            <Author c={c} />
            <p className="mt-0.5 whitespace-pre-wrap text-[13px] leading-relaxed text-text">
              {c.body}
            </p>
          </div>
        ))}
      </div>
      {!root.resolved && (
        <div className="mt-3" onClick={(e) => e.stopPropagation()}>
          <ReplyBox
            placeholder={
              root.author === "dara" ? "Reply to Dara" : "Reply, or write @dara to ask for a change"
            }
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
          className="flex h-6 cursor-pointer items-center gap-1 rounded-md px-1.5 text-[11px] text-text-muted transition-colors hover:bg-surface-raised hover:text-text"
        >
          <Icon name="check" size={12} />
          {root.resolved ? "Reopen" : "Resolve"}
        </button>
        <button
          type="button"
          onClick={() => void remove(root.id)}
          aria-label="Delete thread"
          title="Delete thread"
          className="flex h-6 w-6 cursor-pointer items-center justify-center rounded-md text-text-muted transition-colors hover:bg-danger-subtle hover:text-danger"
        >
          <Icon name="trash-2" size={12} />
        </button>
      </div>
    </li>
  );
};

/** The Comments tab: threads anchored to the document's words, the one being written, and replies that can wake Dara. */
export const DocComments: React.FC<Props> = ({
  draft,
  busy,
  modelLoaded,
  onDraftSubmit,
  onDraftCancel,
  onReply,
  onFocusThread
}) => {
  const items = useCommentsStore((s) => s.items);
  const error = useCommentsStore((s) => s.error);
  const activeId = useCommentsStore((s) => s.activeId);
  const setActive = useCommentsStore((s) => s.setActive);
  const threads = useMemo(() => buildThreads(items), [items]);
  const open = threads.filter((t) => !t.root.resolved);
  const resolved = threads.filter((t) => t.root.resolved);
  const [body, setBody] = useState("");
  const draftRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (draft) {
      setBody("");
      draftRef.current?.focus();
    }
  }, [draft]);

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
          className="rounded-lg border border-danger/40 bg-danger-subtle px-3 py-2 text-xs text-danger"
        >
          {error}
        </p>
      )}

      {draft && (
        <div className="rounded-lg border border-border-strong bg-surface-raised p-3">
          {draft.quote && (
            <p className="mb-2 line-clamp-3 border-l-2 border-primary/60 pl-2 text-xs italic text-text-muted">
              {draft.quote}
            </p>
          )}
          <textarea
            ref={draftRef}
            value={body}
            onChange={(e) => setBody(e.target.value)}
            rows={3}
            placeholder="Write a comment"
            aria-label="New comment"
            className="custom-scrollbar block w-full resize-none rounded-md border border-border bg-bg px-2.5 py-2 text-[13px] text-text outline-none placeholder:text-text-subtle focus:border-border-strong"
          />
          <div className="mt-2 flex items-center justify-end gap-2">
            <Button variant="ghost" size="xs" onClick={onDraftCancel}>
              Cancel
            </Button>
            <Button
              variant="secondary"
              size="xs"
              disabled={!body.trim()}
              onClick={() => onDraftSubmit(body.trim(), false)}
            >
              Comment
            </Button>
            <Button
              variant="primary"
              size="xs"
              disabled={!body.trim() || busy || !modelLoaded}
              onClick={() => onDraftSubmit(body.trim(), true)}
            >
              <Icon name="sparkles" size={12} />
              Ask Dara
            </Button>
          </div>
        </div>
      )}

      {open.length === 0 && !draft && (
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
        <details className="group">
          <summary className="cursor-pointer list-none px-1 text-xs font-semibold text-text-muted hover:text-text">
            Resolved ({resolved.length})
          </summary>
          <ul className="mt-2 space-y-3">{resolved.map(renderCard)}</ul>
        </details>
      )}
    </div>
  );
};
