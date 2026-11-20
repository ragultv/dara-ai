import React, { useEffect, useRef, useState } from "react";
import { Icon } from "../ui/Icon.tsx";
import { Button } from "../ui/Button.tsx";
import { Reasoning } from "../components/assistant-ui/Reasoning.tsx";
import { ThinkingEffort } from "../components/assistant-ui/ThinkingEffort.tsx";
import {
  selectMessages,
  useDocChatStore,
  type AssistantMsg,
  type DocChatMsg
} from "../stores/useDocChatStore.ts";

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
      {msg.steps.length > 0 && (
        <ul className="mb-2 space-y-1.5" aria-label="Progress">
          {msg.steps.map((s, i) => (
            <li
              key={`${i}-${s.label}`}
              className={`flex items-center gap-2 text-xs ${
                s.state === "todo" ? "text-text-subtle" : "text-text-muted"
              }`}
            >
              <StepDot state={s.state} />
              <span className="truncate">{s.label}</span>
            </li>
          ))}
        </ul>
      )}
      {msg.text ? (
        <p className={msg.status === "error" ? "text-danger" : "text-text"}>{msg.text}</p>
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
  onOpenModels
}) => {
  const messages = useDocChatStore(selectMessages(docId));
  const [text, setText] = useState("");
  const listRef = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  useEffect(() => {
    const el = listRef.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [messages]);

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
      <div className="flex h-12 shrink-0 items-center justify-between border-b border-border pl-4 pr-2">
        <p className="text-sm font-semibold text-text">Chat</p>
        <button
          type="button"
          onClick={onClose}
          aria-label="Hide chat"
          title="Hide chat"
          className="flex h-8 w-8 cursor-pointer items-center justify-center rounded-md text-text-muted transition-colors hover:bg-surface-raised hover:text-text"
        >
          <Icon name="x" size={15} />
        </button>
      </div>

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
          messages.map((m) => <Bubble key={m.id} msg={m} onUndo={onUndo} />)
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
              busy ? "Dara is working..." : docIsEmpty ? "Describe a document" : "Describe a change"
            }
            aria-label="Message to Dara"
            className="block max-h-40 w-full resize-none border-0 bg-transparent px-3 pt-2.5 text-[13px] text-text outline-none placeholder:text-text-subtle"
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
    </aside>
  );
};
