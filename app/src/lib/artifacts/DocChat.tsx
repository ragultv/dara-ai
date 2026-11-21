import React, { useEffect, useRef, useState } from "react";
import { Icon } from "../ui/Icon.tsx";
import { Button } from "../ui/Button.tsx";
import { Markdown } from "../ui/Markdown.tsx";
import { Reasoning } from "../components/assistant-ui/Reasoning.tsx";
import { DocHistory } from "./DocHistory.tsx";
import { Timeline } from "../components/assistant-ui/elements/timeline.tsx";
import { ThinkingEffort } from "../components/assistant-ui/ThinkingEffort.tsx";
import {
  selectMessages,
  useDocChatStore,
  type AssistantMsg,
  type DocChatMsg
} from "../stores/useDocChatStore.ts";

export type PanelTab = "chat" | "comments" | "history";

interface Props {
  docId: string;
  docIsEmpty: boolean;
  busy: boolean;
  modelLoaded: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
  onUndo: (msgId: string) => void;
  onClose: () => void;
  onOpenModels: () => void;
  tab: PanelTab;
  onTab: (tab: PanelTab) => void;
  /** the Comments tab content, and how many threads are open (the badge on the tab) */
  commentsSlot: React.ReactNode;
  openThreads: number;
  /** text to put in the message box (and focus it) when the Ask Dara button is used */
  prefill: { text: string; nonce: number } | null;
  updatedAt: number;
  onRestored: (markdown: string) => void;
}

const SUGGEST_EMPTY = [
  "Write a one-page project brief",
  "Draft a meeting agenda with action items",
  "Explain how a hash map works for beginners"
];
const SUGGEST_FILLED = [
  "Add a short summary at the top",
  "Make it more concise",
  "Fix spelling and grammar"
];

const StepDot: React.FC<{ state: "todo" | "active" | "done" }> = ({ state }) =>
  state === "done" ? (
    <Icon name="check" size={13} className="shrink-0 text-text-muted" />
  ) : state === "active" ? (
    <span
      aria-hidden="true"
      className="h-3 w-3 shrink-0 animate-pulse rounded-full border-2 border-primary"
    />
  ) : (
    <span
      aria-hidden="true"
      className="h-3 w-3 shrink-0 rounded-full border border-border-strong"
    />
  );

const Assistant: React.FC<{ msg: AssistantMsg; onUndo: (id: string) => void }> = ({
  msg,
  onUndo
}) => {
  const running = msg.status === "running";
  return (
    <div className="text-left text-[13px] leading-relaxed text-text">
      {(msg.reasoning || msg.reasoningStreaming) && (
        <Reasoning
          text={msg.reasoning}
          streaming={msg.reasoningStreaming}
          durationMs={msg.thinkingMs}
        />
      )}
      {/* the task as a timeline; Dara's answer is written only when the task is finished */}
      {msg.steps.length > 0 && !msg.reasoningStreaming && (
        <Timeline
          aria-label="Progress"
          className="mb-3"
          visibleCount={msg.steps.length}
          events={msg.steps.map((s, i) => ({
            id: String(i),
            when: s.state === "done" ? "past" : s.state === "active" ? "now" : "future",
            time: s.time ?? "",
            title: s.label,
            detail: s.detail
          }))}
        />
      )}
      {msg.text ? (
        msg.status === "error" ? (
          <p className="text-danger">{msg.text}</p>
        ) : (
          <div className="doc-reply text-[13px] leading-relaxed text-text">
            <Markdown content={msg.text} />
          </div>
        )
      ) : (
        running &&
        msg.steps.length === 0 &&
        !msg.reasoning && <p className="reasoning-shimmer text-text-muted">Working on it...</p>
      )}
      {msg.undo && (
        <button
          type="button"
          onClick={() => onUndo(msg.id)}
          className="mt-2 flex cursor-pointer items-center gap-1.5 rounded-md border border-border px-2 py-1 text-xs font-semibold text-text-muted transition-colors hover:bg-surface-raised hover:text-text"
        >
          <Icon name="rotate-ccw" size={12} />
          Undo
        </button>
      )}
    </div>
  );
};

const Bubble: React.FC<{ msg: DocChatMsg; onUndo: (id: string) => void }> = ({ msg, onUndo }) =>
  msg.role === "user" ? (
    <div className="flex justify-end">
      <p className="max-w-[88%] whitespace-pre-wrap rounded-lg bg-surface-raised px-3 py-2 text-left text-[13px] leading-relaxed text-text">
        {msg.text}
      </p>
    </div>
  ) : (
    <Assistant msg={msg} onUndo={onUndo} />
  );

/** The conversation beside a document: ask for a change, watch Dara plan and write, undo it. */
export const DocChat: React.FC<Props> = ({
  docId,
  docIsEmpty,
  busy,
  modelLoaded,
  onSend,
  onStop,
  onUndo,
  onClose,
  onOpenModels,
  tab,
  onTab,
  commentsSlot,
  openThreads,
  prefill,
  updatedAt,
  onRestored
}) => {
  const messages = useDocChatStore(selectMessages(docId));
  const clearChat = useDocChatStore((s) => s.clear);
  const saveProblem = useDocChatStore((s) => s.saveProblem[docId]);

  // the chat belongs to the document: read what was saved with it
  useEffect(() => {
    void useDocChatStore.getState().load(docId);
  }, [docId]);
  const [text, setText] = useState("");
  const listRef = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = listRef.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [messages]);

  // the Ask Dara button puts words in the box (the selected text) and moves the cursor there
  useEffect(() => {
    if (!prefill) return;
    setText(prefill.text);
    requestAnimationFrame(() => {
      const el = inputRef.current;
      if (!el) return;
      el.focus();
      el.setSelectionRange(el.value.length, el.value.length);
    });
  }, [prefill]);

  // the box grows with what is typed, up to a limit, then scrolls
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [text]);

  const submit = (value: string) => {
    const v = value.trim();
    if (!v || busy || !modelLoaded) return;
    setText("");
    stick.current = true;
    onSend(v);
  };

  return (
    <aside
      aria-label="Chat with Dara about this document"
      className="flex h-full w-[360px] shrink-0 flex-col border-l border-border bg-surface"
    >
      <div className="flex h-12 shrink-0 items-center justify-between border-b border-border pl-3 pr-2">
        <div role="tablist" aria-label="Panel" className="flex items-center gap-1">
          {(["chat", "comments", "history"] as const).map((t) => (
            <button
              key={t}
              type="button"
              role="tab"
              aria-selected={tab === t}
              onClick={() => onTab(t)}
              className={`h-8 cursor-pointer rounded-md px-2.5 text-sm font-semibold transition-colors ${
                tab === t
                  ? "bg-surface-raised text-text"
                  : "text-text-muted hover:bg-surface-raised hover:text-text"
              }`}
            >
              {t === "chat" ? "Chat" : t === "comments" ? "Comments" : "History"}
              {t === "comments" && openThreads > 0 && (
                <span className="ml-1.5 rounded-full bg-primary px-1.5 py-px text-[10px] font-semibold text-white">
                  {openThreads}
                </span>
              )}
            </button>
          ))}
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Hide panel"
          title="Hide panel"
          className="flex h-8 w-8 cursor-pointer items-center justify-center rounded-md text-text-muted transition-colors hover:bg-surface-raised hover:text-text"
        >
          <Icon name="x" size={15} />
        </button>
      </div>

      {tab === "comments" && commentsSlot}

      {tab === "history" && (
        <DocHistory docId={docId} updatedAt={updatedAt} onRestored={onRestored} />
      )}

      {/* the chat stays mounted (hidden) while History is shown, so a half-typed message is kept */}
      <div className={tab === "chat" ? "flex min-h-0 flex-1 flex-col" : "hidden"}>
        <div
          ref={listRef}
          onScroll={() => {
            const el = listRef.current;
            if (el) stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
          }}
          className="min-h-0 flex-1 space-y-4 overflow-y-auto px-4 py-4 custom-scrollbar"
        >
          {messages.length === 0 ? (
            <div className="flex flex-col gap-3 pt-2">
              <p className="text-xs text-text-muted">
                {docIsEmpty
                  ? "Describe a document. Dara plans it, then writes it one section at a time."
                  : "Ask for a change. Dara edits only the parts you mention and leaves the rest as it is."}
              </p>
              <div className="flex flex-col items-start gap-2">
                {(docIsEmpty ? SUGGEST_EMPTY : SUGGEST_FILLED).map((s) => (
                  <button
                    key={s}
                    type="button"
                    disabled={!modelLoaded}
                    onClick={() => submit(s)}
                    className="cursor-pointer rounded-lg border border-border px-3 py-1.5 text-left text-xs text-text-muted transition-colors hover:border-border-strong hover:text-text disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <>
              {messages.map((m) => (
                <Bubble key={m.id} msg={m} onUndo={onUndo} />
              ))}
              {!busy && (
                <div className="flex justify-center pt-1">
                  <button
                    type="button"
                    onClick={() => void clearChat(docId)}
                    className="cursor-pointer text-[11px] text-text-subtle underline-offset-2 hover:text-text hover:underline"
                  >
                    Clear chat
                  </button>
                </div>
              )}
            </>
          )}
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            submit(text);
          }}
          className="shrink-0 border-t border-border p-3"
        >
          {!modelLoaded && (
            <div className="mb-2 flex items-center justify-between gap-3 rounded-lg border border-border px-3 py-2 text-xs text-text-muted">
              <span>Load a model to chat with Dara.</span>
              <Button type="button" variant="secondary" size="xs" onClick={onOpenModels}>
                Models
              </Button>
            </div>
          )}
          <div className="rounded-lg border border-border bg-bg focus-within:border-border-strong">
            <textarea
              ref={inputRef}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  submit(text);
                }
              }}
              rows={2}
              disabled={busy}
              placeholder={
                busy
                  ? "Dara is working..."
                  : docIsEmpty
                    ? "Describe a document"
                    : "Describe a change"
              }
              aria-label="Message to Dara"
              className="custom-scrollbar block max-h-40 w-full resize-none border-0 bg-transparent px-3 pt-2.5 text-[13px] text-text outline-none placeholder:text-text-subtle"
            />
            <div className="flex items-center justify-between px-2 pb-2 pt-1">
              <ThinkingEffort />
              {busy ? (
                <button
                  type="button"
                  onClick={onStop}
                  aria-label="Stop"
                  title="Stop"
                  className="flex h-7 w-7 cursor-pointer items-center justify-center rounded-md bg-primary text-white transition-colors hover:bg-primary-hover"
                >
                  <Icon name="square" size={12} />
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={!text.trim() || !modelLoaded}
                  aria-label="Send"
                  title="Send"
                  className="flex h-7 w-7 cursor-pointer items-center justify-center rounded-md bg-primary text-white transition-colors hover:bg-primary-hover disabled:cursor-not-allowed disabled:opacity-40"
                >
                  <Icon name="send" size={13} />
                </button>
              )}
            </div>
          </div>
        </form>
      </div>
    </aside>
  );
};
