import React, { useState } from "react";
import { Icon } from "../../ui/Icon.tsx";
import { Markdown } from "../../ui/Markdown.tsx";
import { MessageActions } from "./MessageActions.tsx";
import type { UiMessage } from "../../chat/types.ts";
import { Logo } from "../../ui/Logo.tsx";
import { Reasoning } from "./Reasoning.tsx";
import { ToolCalls } from "./ToolCalls.tsx";
import { displayText, extractBlocks } from "../../chat/tools/protocol.ts";
import type { ToolResultItem } from "../../chat/tools/types.ts";

interface MessagePairProps {
  message: UiMessage;
  /** Needed to show the reply's tool calls with their live state. */
  conversationId?: string;
  /** Results of this reply's tool calls (from the tool-results message after it), when there is one. */
  toolResults?: ToolResultItem[] | null;
  isStreaming?: boolean;
  /** The newest assistant message: only it offers Continue. */
  isLast?: boolean;
  onEdit?: (newText: string) => void;
  onRegenerate?: () => void;
  onContinue?: () => void;
  /** Offered when the engine says image support is not loaded. */
  onSetupVision?: () => void;
}

export const MessagePair: React.FC<MessagePairProps> = ({
  message,
  conversationId,
  toolResults = null,
  isStreaming = false,
  isLast = false,
  onEdit,
  onRegenerate,
  onContinue,
  onSetupVision
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editText, setEditText] = useState(message.content);
  const [copied, setCopied] = useState(false);

  const handleCopyUser = () => {
    void navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleSaveEdit = () => {
    if (!editText.trim()) return;
    setIsEditing(false);
    onEdit?.(editText);
  };

  if (message.role === "user") {
    return (
      <div className="flex flex-col items-end group">
        {isEditing ? (
          <div className="w-full max-w-2xl rounded-2xl bg-surface-raised border border-border focus-within:border-border-strong shadow-md p-3.5 space-y-3">
            <textarea
              value={editText}
              onChange={(e) => setEditText(e.target.value)}
              rows={3}
              aria-label="Edit your message"
              className="w-full bg-transparent p-0 text-sm text-text placeholder:text-text-subtle resize-none border-0 outline-none ring-0 leading-relaxed"
              style={{ outline: "none", boxShadow: "none", border: "none" }}
              placeholder="Edit your message..."
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSaveEdit();
                } else if (e.key === "Escape") {
                  setIsEditing(false);
                }
              }}
            />
            <div className="flex items-center justify-end gap-2 pt-2 border-t border-border/40">
              <button
                type="button"
                onClick={() => setIsEditing(false)}
                className="px-3 py-1.5 rounded-lg border border-border bg-surface hover:bg-surface-raised text-xs font-semibold text-text-muted hover:text-text transition-colors cursor-pointer"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleSaveEdit}
                className="px-3.5 py-1.5 rounded-lg bg-primary hover:bg-primary-hover text-white text-xs font-semibold transition-all cursor-pointer shadow-sm"
              >
                Save & resend
              </button>
            </div>
          </div>
        ) : (
          <>
            <div className="max-w-[75%] rounded-2xl rounded-tr-sm px-4 py-2.5 bg-surface-raised border border-border text-sm text-text space-y-2 shadow-xs">
              {message.images && message.images.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {message.images.map((img, i) => (
                    <img
                      key={i}
                      src={img}
                      alt="Attachment"
                      className="max-w-xs max-h-48 rounded-lg border border-border object-cover"
                    />
                  ))}
                </div>
              )}
              {message.images === undefined && (message.attachmentIds?.length ?? 0) > 0 && (
                <div className="text-[11px] text-text-subtle italic">
                  {message.attachmentIds?.length} image
                  {message.attachmentIds?.length === 1 ? "" : "s"} (loading...)
                </div>
              )}
              {message.content && (
                <p className="leading-relaxed whitespace-pre-wrap break-words">{message.content}</p>
              )}
            </div>

            <div className="opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity flex items-center gap-1 mt-1 pr-1">
              <button
                type="button"
                aria-label="Edit message"
                onClick={() => {
                  setEditText(message.content);
                  setIsEditing(true);
                }}
                className="p-1 rounded-md text-text-subtle hover:text-text hover:bg-surface-raised transition-colors cursor-pointer"
                title="Edit message"
              >
                <Icon name="pencil" size={13} />
              </button>
              <button
                type="button"
                aria-label="Copy message"
                onClick={handleCopyUser}
                className="p-1 rounded-md text-text-subtle hover:text-text hover:bg-surface-raised transition-colors cursor-pointer"
                title={copied ? "Copied!" : "Copy message"}
              >
                {copied ? (
                  <Icon name="check" size={13} className="text-success" />
                ) : (
                  <Icon name="copy" size={13} />
                )}
              </button>
            </div>
          </>
        )}
      </div>
    );
  }

  // Assistant message: the text without tool_call blocks, the calls shown as rows
  const shown = displayText(message.content);
  const hasCalls = conversationId !== undefined && extractBlocks(message.content).length > 0;
  if (!isStreaming && !shown && !hasCalls && !message.error && !message.reasoning?.trim()) {
    return null; // no empty phantom bubbles
  }

  const hasError = Boolean(message.error);
  const hasText = shown.length > 0;
  const hasReasoning = Boolean(message.reasoning?.trim());
  // the thought is still being written while the reply streams and no answer text has started
  const thinkingNow = isStreaming && hasReasoning && !hasText;
  const hitLimit = message.finishReason === "length" && !isStreaming && !hasError;

  return (
    <div className="flex gap-3 group text-left">
      <Logo
        alt="Dara AI"
        className={`w-7 h-7 object-contain shrink-0 ${isStreaming ? "animate-pulse" : ""}`}
      />
      <div className="flex-1 min-w-0 space-y-1">
        {hasReasoning && (
          <Reasoning
            text={message.reasoning ?? ""}
            streaming={thinkingNow}
            durationMs={message.reasoningMs}
            cut={message.reasoningCut}
          />
        )}

        {isStreaming && !hasText && !hasReasoning && !hasCalls && (
          <div className="flex items-center gap-2 h-7 text-text-subtle">
            <span className="w-1.5 h-1.5 rounded-full bg-text-muted animate-pulse" />
            <span className="text-xs text-text-muted font-medium">Generating response...</span>
          </div>
        )}

        {hasText && (
          <div className="text-sm leading-relaxed text-text">
            <Markdown content={shown} />
          </div>
        )}

        {hasCalls && conversationId !== undefined && (
          <ToolCalls
            conversationId={conversationId}
            assistantMessageId={message.id}
            content={message.content}
            results={toolResults}
            streaming={isStreaming}
          />
        )}

        {hasError && (
          <div
            role="alert"
            className="rounded-xl border border-danger/40 bg-danger/10 p-3.5 space-y-2 text-xs"
          >
            <div className="flex items-center gap-2 text-danger font-semibold">
              <Icon name="alert-triangle" size={14} />
              <span>
                {message.errorCode === "vision_unavailable"
                  ? "Image support is not loaded"
                  : "Generation error"}
              </span>
            </div>
            <p className="text-text-muted leading-relaxed select-text text-[12px] whitespace-pre-wrap">
              {message.error}
            </p>
            <div className="pt-1 flex flex-wrap gap-2">
              {message.errorCode === "vision_unavailable" && onSetupVision && (
                <button
                  type="button"
                  onClick={onSetupVision}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-primary hover:bg-primary-hover text-white text-xs font-semibold transition-all cursor-pointer shadow-xs"
                >
                  <Icon name="image-plus" size={12} />
                  <span>Turn on image support</span>
                </button>
              )}
              {onRegenerate && (
                <button
                  type="button"
                  onClick={onRegenerate}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-danger/40 bg-surface hover:bg-surface-raised text-xs font-semibold text-text transition-all cursor-pointer shadow-xs"
                >
                  <Icon name="rotate-ccw" size={12} className="text-danger" />
                  <span>Try again</span>
                </button>
              )}
            </div>
          </div>
        )}

        {hitLimit && (
          <div className="flex flex-wrap items-center gap-2 pt-1 text-[11px] text-text-subtle">
            <Icon name="info" size={11} />
            <span>The reply reached its length limit or the end of the context.</span>
            {isLast && onContinue && (
              <button
                type="button"
                onClick={onContinue}
                className="px-2.5 py-1 rounded-lg border border-border bg-surface hover:bg-surface-raised text-[11px] font-semibold text-text transition-colors cursor-pointer"
              >
                Continue
              </button>
            )}
          </div>
        )}

        {message.interrupted && (
          <div className="inline-flex items-center gap-1 text-[11px] text-text-subtle italic pt-1">
            <Icon name="square" size={10} />
            <span>Generation stopped</span>
          </div>
        )}

        {!isStreaming && (hasText || hasReasoning) && !hasError && (
          <div className="pt-1">
            <MessageActions content={shown} stats={message.stats} onRegenerate={onRegenerate} />
          </div>
        )}
      </div>
    </div>
  );
};
