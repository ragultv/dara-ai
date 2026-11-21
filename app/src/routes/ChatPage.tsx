import React, { useRef, useEffect } from "react";
import { useChatStore, type Conversation } from "../lib/stores/useChatStore.ts";
import { useModelsStore } from "../lib/stores/useModelsStore.ts";
import { useServerStore } from "../lib/stores/useServerStore.ts";
import { Icon } from "../lib/ui/Icon.tsx";
import { Composer } from "../lib/components/assistant-ui/Composer.tsx";
import { MessagePair } from "../lib/components/assistant-ui/MessagePair.tsx";
import { toast } from "../lib/ui/Toast.tsx";
import { enableAddon } from "../lib/chat/addons.ts";
import type { ContextWindow } from "../lib/chat/types.ts";
import { Logo } from "../lib/ui/Logo.tsx";
import { isToolResultsContent, parseResults } from "../lib/chat/tools/protocol.ts";
import { ToolApprovalDialog } from "../lib/components/assistant-ui/ToolApprovalDialog.tsx";

function groupConversations(convs: Conversation[]) {
  const now = new Date();
  const today: Conversation[] = [];
  const yesterday: Conversation[] = [];
  const older: Conversation[] = [];
  const dayMs = 86400000;

  for (const c of convs) {
    const time = new Date(c.updatedAt).getTime();
    const diff = now.getTime() - time;
    if (diff < dayMs) today.push(c);
    else if (diff < dayMs * 2) yesterday.push(c);
    else older.push(c);
  }
  return { today, yesterday, older };
}

export const ChatPage: React.FC = () => {
  const conversations = useChatStore((s) => s.conversations);
  const activeConversationId = useChatStore((s) => s.activeConversationId);
  const isGenerating = useChatStore((s) => s.isGenerating);
  const streamingText = useChatStore((s) => s.streamingText);
  const createConversation = useChatStore((s) => s.createConversation);
  const selectConversation = useChatStore((s) => s.selectConversation);
  const deleteConversation = useChatStore((s) => s.deleteConversation);
  const editAndResend = useChatStore((s) => s.editAndResend);
  const regenerate = useChatStore((s) => s.regenerate);
  const getActiveConversation = useChatStore((s) => s.getActiveConversation);
  const sendMessage = useChatStore((s) => s.sendMessage);
  const continueGeneration = useChatStore((s) => s.continueGeneration);
  const phase = useChatStore((s) => s.phase);

  const activeConv = getActiveConversation();
  const loadedModel = useModelsStore((s) => s.models.find((m) => m.dara.loaded) ?? null);
  const serverLoaded = useServerStore((s) => s.loadedModel);

  const setupVisionThenRetry = async (messageId: string) => {
    if (!loadedModel) return;
    try {
      await enableAddon(loadedModel.id, "vision");
      toast.show("Image support is on. Retrying.", "success");
      regenerate(messageId, loadedModel.id);
    } catch (err) {
      toast.show(err instanceof Error ? err.message : "Could not set up image support", "error");
    }
  };

  const messagesViewportRef = useRef<HTMLDivElement | null>(null);

  // Auto-scroll messages viewport to bottom on stream changes
  useEffect(() => {
    if (!messagesViewportRef.current) return;
    const viewport = messagesViewportRef.current;
    const distFromBottom = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight;
    if (distFromBottom < 300 || isGenerating) {
      viewport.scrollTop = viewport.scrollHeight;
    }
  }, [streamingText, isGenerating, activeConv?.messages.length]);

  const grouped = groupConversations(conversations);

  const handlePromptSuggestion = (prompt: string) => {
    void sendMessage(prompt);
  };

  return (
    <div className="flex h-full w-full overflow-hidden bg-bg text-text select-text">
      {/* Threads Drawer / Sidebar */}
      <aside className="w-64 border-r border-border bg-bg-subtle flex flex-col shrink-0 select-none">
        <div className="p-3 border-b border-border">
          <button
            type="button"
            onClick={() => createConversation()}
            className="w-full flex items-center justify-between px-3.5 py-2 rounded-xl bg-primary hover:bg-primary-hover text-white text-xs font-semibold shadow-xs transition-all cursor-pointer"
          >
            <span>New Chat</span>
            <Icon name="plus" size={14} />
          </button>
        </div>

        {/* Conversation List */}
        <div className="flex-1 overflow-y-auto p-2 space-y-4 custom-scrollbar text-left">
          {conversations.length === 0 ? (
            <div className="p-4 text-center text-xs text-text-subtle italic">No chats yet</div>
          ) : (
            <>
              {grouped.today.length > 0 && (
                <div className="space-y-1">
                  <span className="px-2 text-[10px] font-bold text-text-subtle uppercase tracking-wider">
                    Today
                  </span>
                  {grouped.today.map((c) => (
                    <ConversationItem
                      key={c.id}
                      conv={c}
                      isActive={c.id === activeConversationId}
                      onSelect={() => selectConversation(c.id)}
                      onDelete={() => deleteConversation(c.id)}
                    />
                  ))}
                </div>
              )}

              {grouped.yesterday.length > 0 && (
                <div className="space-y-1">
                  <span className="px-2 text-[10px] font-bold text-text-subtle uppercase tracking-wider">
                    Yesterday
                  </span>
                  {grouped.yesterday.map((c) => (
                    <ConversationItem
                      key={c.id}
                      conv={c}
                      isActive={c.id === activeConversationId}
                      onSelect={() => selectConversation(c.id)}
                      onDelete={() => deleteConversation(c.id)}
                    />
                  ))}
                </div>
              )}

              {grouped.older.length > 0 && (
                <div className="space-y-1">
                  <span className="px-2 text-[10px] font-bold text-text-subtle uppercase tracking-wider">
                    Older
                  </span>
                  {grouped.older.map((c) => (
                    <ConversationItem
                      key={c.id}
                      conv={c}
                      isActive={c.id === activeConversationId}
                      onSelect={() => selectConversation(c.id)}
                      onDelete={() => deleteConversation(c.id)}
                    />
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      </aside>

      {/* Main Chat Viewport */}
      <main className="flex-1 flex flex-col h-full min-w-0 bg-bg overflow-hidden relative">
        {/* Messages Scroll Area */}
        <div
          ref={messagesViewportRef}
          className="flex-1 overflow-y-auto px-4 sm:px-8 py-6 space-y-6 custom-scrollbar select-text"
        >
          {!activeConv || activeConv.messages.length === 0 ? (
            /* Empty State Hero */
            <div className="h-full flex flex-col items-center justify-center max-w-xl mx-auto text-center space-y-6 p-4">
              <div className="flex flex-col items-center space-y-2">
                <Logo alt="Dara" className="w-14 h-14 object-contain" />
                <h2 className="text-xl font-bold tracking-tight text-text">Dara Studio</h2>
                <p className="text-xs text-text-muted leading-relaxed max-w-sm">
                  High-throughput offline AI running directly on your GPU with flash drafting and
                  dynamic context scaling.
                </p>
              </div>

              {/* Starter Suggestions */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 w-full pt-2">
                {[
                  "Explain quantum entanglement in plain English",
                  "Write a fast Python merge sort algorithm",
                  "Optimize a high-throughput SQL query",
                  "Explain model quantization (INT4 vs INT8)"
                ].map((prompt, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => handlePromptSuggestion(prompt)}
                    className="p-3 text-left rounded-xl border border-border bg-surface hover:bg-surface-raised text-xs text-text-muted hover:text-text transition-all cursor-pointer shadow-xs leading-relaxed"
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            /* Messages List - Exactly ONE render per message, zero duplicate streaming bubbles */
            <div className="max-w-3xl mx-auto space-y-6 w-full pb-4">
              {activeConv.messages.map((m, index) => {
                const windowHere = activeConv.windows.find((w) => w.startMessageId === m.id);
                // tool results are messages for the model: shown inside the reply that asked for them, not as a bubble
                if (m.role === "user" && isToolResultsContent(m.content))
                  return windowHere ? (
                    <ContextWindowDivider key={m.id} window={windowHere} />
                  ) : null;
                const isLastMessage = index === activeConv.messages.length - 1;
                const isLastAssistant =
                  isGenerating && phase === "generating" && isLastMessage && m.role === "assistant";
                const next = activeConv.messages[index + 1];
                const toolResults =
                  m.role === "assistant" &&
                  next?.role === "user" &&
                  isToolResultsContent(next.content)
                    ? parseResults(next.content)
                    : null;

                return (
                  <React.Fragment key={m.id}>
                    {windowHere && <ContextWindowDivider window={windowHere} />}
                    <MessagePair
                      conversationId={activeConv.id}
                      toolResults={toolResults}
                      message={m}
                      isStreaming={isLastAssistant}
                      isLast={isLastMessage}
                      onEdit={(newText) => editAndResend(m.id, newText, serverLoaded ?? undefined)}
                      onRegenerate={() => regenerate(m.id, serverLoaded ?? undefined)}
                      onContinue={() => continueGeneration(serverLoaded ?? undefined)}
                      onSetupVision={() => void setupVisionThenRetry(m.id)}
                    />
                  </React.Fragment>
                );
              })}
            </div>
          )}
        </div>

        {/* Assistant-UI Unified Composer */}
        <Composer />
      </main>

      {/* Asks before a tool call runs; shows the first call waiting (docs/CHAT_TOOLS.md) */}
      <ToolApprovalDialog />
    </div>
  );
};

/** Marks where a new context window starts. The messages above stay saved and visible; the model sees the summary instead. */
const ContextWindowDivider: React.FC<{ window: ContextWindow }> = ({ window: win }) => {
  const [open, setOpen] = React.useState(false);
  return (
    <div className="py-1">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="w-full flex items-center gap-3 text-[11px] text-text-subtle hover:text-text cursor-pointer"
      >
        <span className="flex-1 h-px bg-border" />
        <span className="inline-flex items-center gap-1.5 font-semibold uppercase tracking-wider">
          <Icon name="layers" size={12} />
          New context window · {win.summarizedMessages} earlier message
          {win.summarizedMessages === 1 ? "" : "s"} summarized
          <Icon name={open ? "chevron-up" : "chevron-down"} size={11} />
        </span>
        <span className="flex-1 h-px bg-border" />
      </button>
      {open && (
        <div className="mt-2 p-3 rounded-xl border border-border bg-surface text-xs text-text-muted leading-relaxed whitespace-pre-wrap select-text">
          <div className="text-[10px] font-bold uppercase tracking-wider text-text-subtle mb-1">
            What the model remembers of the messages above
          </div>
          {win.summary}
        </div>
      )}
    </div>
  );
};

const ConversationItem: React.FC<{
  conv: Conversation;
  isActive: boolean;
  onSelect: () => void;
  onDelete: () => void;
}> = ({ conv, isActive, onSelect, onDelete }) => {
  return (
    <div
      onClick={onSelect}
      className={`group flex items-center justify-between px-2.5 py-2 rounded-lg text-xs cursor-pointer transition-colors border ${
        isActive
          ? "bg-surface-raised text-text font-semibold border-transparent"
          : "border-transparent text-text-muted hover:text-text hover:bg-surface-raised/60"
      }`}
    >
      <div className="flex items-center gap-2 truncate min-w-0">
        <Icon
          name="message-square"
          size={13}
          className={isActive ? "text-text" : "text-text-subtle"}
        />
        <span className="truncate">{conv.title || "New Chat"}</span>
      </div>

      <button
        type="button"
        aria-label="Delete chat"
        onClick={(e) => {
          e.stopPropagation();
          onDelete();
        }}
        className="opacity-0 group-hover:opacity-100 p-1 rounded-md text-text-subtle hover:text-danger hover:bg-surface transition-all cursor-pointer"
      >
        <Icon name="trash-2" size={12} />
      </button>
    </div>
  );
};
