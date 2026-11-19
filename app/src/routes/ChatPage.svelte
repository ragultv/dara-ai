<script lang="ts">
  import { chatStore, type UiMessage, type Conversation } from "../lib/stores/chat.svelte.ts";
  import { serverStore } from "../lib/stores/server.svelte.ts";
  import { modelsStore } from "../lib/stores/models.svelte.ts";
  import Icon from "../lib/ui/Icon.svelte";
  import Markdown from "../lib/ui/Markdown.svelte";
  import Toggle from "../lib/ui/Toggle.svelte";
  import Slider from "../lib/ui/Slider.svelte";
  import Textarea from "../lib/ui/Textarea.svelte";
  import Button from "../lib/ui/Button.svelte";
  import Popover from "../lib/ui/Popover.svelte";
  import { toast } from "../lib/ui/Toast.svelte";
  import NumberTicker from "../lib/ui/NumberTicker.svelte";
  import MessageActions from "../lib/ui/MessageActions.svelte";
  import ContextBreakdown from "../lib/ui/ContextBreakdown.svelte";

  let inputText = $state("");
  let attachedImages = $state<string[]>([]);
  let fileInputRef: HTMLInputElement | null = $state(null);
  let textareaRef: HTMLTextAreaElement | null = $state(null);
  let messagesViewportRef: HTMLDivElement | null = $state(null);
  let isEditingId = $state<string | null>(null);
  let editText = $state("");
  let isContextPopoverOpen = $state(false);
  let copiedMessageId = $state<string | null>(null);

  const activeConv = $derived(chatStore.activeConversation);
  const loadedModelInfo = $derived(modelsStore.loadedModel);
  const isVisionSupported = $derived(loadedModelInfo?.dara.vision?.active ?? true);

  // Auto-scroll messages viewport to bottom as tokens stream in
  $effect(() => {
    const messages = activeConv?.messages;
    if (messages && messages.length > 0) {
      const last = messages[messages.length - 1];
      const _ = last?.content; // Track content changes during streaming
      requestAnimationFrame(() => {
        if (messagesViewportRef) {
          const distFromBottom =
            messagesViewportRef.scrollHeight -
            messagesViewportRef.scrollTop -
            messagesViewportRef.clientHeight;
          if (distFromBottom < 250 || chatStore.isGenerating) {
            messagesViewportRef.scrollTop = messagesViewportRef.scrollHeight;
          }
        }
      });
    }
  });

  function autoResizeTextarea() {
    if (!textareaRef) return;
    textareaRef.style.height = "auto";
    textareaRef.style.height = `${Math.min(Math.max(textareaRef.scrollHeight, 44), 160)}px`;
  }

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

  const grouped = $derived(groupConversations(chatStore.conversations));

  function handleSend() {
    if ((!inputText.trim() && attachedImages.length === 0) || chatStore.isGenerating) return;
    const text = inputText;
    const imgs = [...attachedImages];
    inputText = "";
    attachedImages = [];
    if (textareaRef) {
      textareaRef.style.height = "auto";
    }
    void chatStore.sendMessage(text, imgs, serverStore.loadedModel ?? undefined);
  }

  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function handleFileSelect(e: Event & { currentTarget: HTMLInputElement }) {
    const files = e.currentTarget.files;
    if (!files || files.length === 0) return;

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      if (!file) continue;
      const reader = new FileReader();
      reader.onload = (event) => {
        const result = event.target?.result;
        if (typeof result === "string") {
          attachedImages = [...attachedImages, result];
        }
      };
      reader.readAsDataURL(file);
    }
    e.currentTarget.value = "";
  }

  function copyMessage(text: string, msgId?: string) {
    void navigator.clipboard.writeText(text);
    if (msgId) {
      copiedMessageId = msgId;
      setTimeout(() => {
        if (copiedMessageId === msgId) copiedMessageId = null;
      }, 2000);
    }
    toast.show("Copied to clipboard.", "info");
  }

  function startEdit(msg: UiMessage) {
    isEditingId = msg.id;
    editText = msg.content;
  }

  function submitEdit(msgId: string) {
    if (!editText.trim()) return;
    const text = editText;
    isEditingId = null;
    chatStore.editAndResend(msgId, text, serverStore.loadedModel ?? undefined);
  }

  const contextPercent = $derived.by(() => {
    if (!activeConv?.context || activeConv.context.capacity === 0) return 0;
    return Math.min(100, Math.round((activeConv.context.used / activeConv.context.capacity) * 100));
  });
</script>

<div class="flex-1 flex h-full overflow-hidden text-left bg-bg">
  <!-- 1. Left: Conversations Rail (Oprel Sidebar Style) -->
  <aside
    class="w-64 border-r border-border bg-bg-subtle flex flex-col justify-between shrink-0 select-none overflow-hidden"
    aria-label="Conversations"
  >
    <div class="p-3 space-y-3 flex-1 flex flex-col overflow-hidden">
      <!-- New Chat Button -->
      <button
        type="button"
        onclick={() => chatStore.createConversation()}
        class="w-full flex items-center justify-between px-3.5 py-2.5 rounded-lg bg-primary hover:bg-primary-hover text-white text-xs font-semibold transition-all cursor-pointer shadow-sm"
      >
        <span>New Chat</span>
        <Icon name="message-square-plus" size={15} />
      </button>

      <!-- Thread List Grouped by Today / Yesterday / Older -->
      <div class="flex-1 overflow-y-auto space-y-4 pr-1">
        {#if chatStore.conversations.length === 0}
          <div class="text-center py-8 text-text-subtle text-xs italic">No chats yet</div>
        {:else}
          {#if grouped.today.length > 0}
            <div class="space-y-1">
              <div class="px-2 text-[10px] font-bold text-text-subtle uppercase tracking-wider">Today</div>
              {#each grouped.today as conv (conv.id)}
                {@const isActive = chatStore.activeConversationId === conv.id}
                <div
                  role="button"
                  tabindex="0"
                  onclick={() => chatStore.selectConversation(conv.id)}
                  onkeydown={(e) => {
                    if (e.key === "Enter" || e.key === " ") chatStore.selectConversation(conv.id);
                  }}
                  class="group flex items-center justify-between px-2.5 py-2 rounded-lg text-xs cursor-pointer transition-colors border {isActive
                    ? 'bg-surface-raised border-primary/30 text-text font-bold'
                    : 'border-transparent text-text-muted hover:text-text hover:bg-surface-raised/60'}"
                >
                  <span class="truncate flex-1">{conv.title}</span>
                  <button
                    type="button"
                    aria-label="Delete chat"
                    onclick={(e) => {
                      e.stopPropagation();
                      chatStore.deleteConversation(conv.id);
                    }}
                    class="opacity-0 group-hover:opacity-100 p-1 text-text-subtle hover:text-danger rounded transition-opacity"
                  >
                    <Icon name="trash-2" size={12} />
                  </button>
                </div>
              {/each}
            </div>
          {/if}

          {#if grouped.yesterday.length > 0}
            <div class="space-y-1">
              <div class="px-2 text-[10px] font-bold text-text-subtle uppercase tracking-wider">Yesterday</div>
              {#each grouped.yesterday as conv (conv.id)}
                {@const isActive = chatStore.activeConversationId === conv.id}
                <div
                  role="button"
                  tabindex="0"
                  onclick={() => chatStore.selectConversation(conv.id)}
                  onkeydown={(e) => {
                    if (e.key === "Enter" || e.key === " ") chatStore.selectConversation(conv.id);
                  }}
                  class="group flex items-center justify-between px-2.5 py-2 rounded-lg text-xs cursor-pointer transition-colors border {isActive
                    ? 'bg-surface-raised border-primary/30 text-text font-bold'
                    : 'border-transparent text-text-muted hover:text-text hover:bg-surface-raised/60'}"
                >
                  <span class="truncate flex-1">{conv.title}</span>
                  <button
                    type="button"
                    aria-label="Delete chat"
                    onclick={(e) => {
                      e.stopPropagation();
                      chatStore.deleteConversation(conv.id);
                    }}
                    class="opacity-0 group-hover:opacity-100 p-1 text-text-subtle hover:text-danger rounded transition-opacity"
                  >
                    <Icon name="trash-2" size={12} />
                  </button>
                </div>
              {/each}
            </div>
          {/if}

          {#if grouped.older.length > 0}
            <div class="space-y-1">
              <div class="px-2 text-[10px] font-bold text-text-subtle uppercase tracking-wider">Older</div>
              {#each grouped.older as conv (conv.id)}
                {@const isActive = chatStore.activeConversationId === conv.id}
                <div
                  role="button"
                  tabindex="0"
                  onclick={() => chatStore.selectConversation(conv.id)}
                  onkeydown={(e) => {
                    if (e.key === "Enter" || e.key === " ") chatStore.selectConversation(conv.id);
                  }}
                  class="group flex items-center justify-between px-2.5 py-2 rounded-lg text-xs cursor-pointer transition-colors border {isActive
                    ? 'bg-surface-raised border-primary/30 text-text font-bold'
                    : 'border-transparent text-text-muted hover:text-text hover:bg-surface-raised/60'}"
                >
                  <span class="truncate flex-1">{conv.title}</span>
                  <button
                    type="button"
                    aria-label="Delete chat"
                    onclick={(e) => {
                      e.stopPropagation();
                      chatStore.deleteConversation(conv.id);
                    }}
                    class="opacity-0 group-hover:opacity-100 p-1 text-text-subtle hover:text-danger rounded transition-opacity"
                  >
                    <Icon name="trash-2" size={12} />
                  </button>
                </div>
              {/each}
            </div>
          {/if}
        {/if}
      </div>
    </div>
  </aside>

  <!-- 2. Center: Chat Messages & Prompt Area -->
  <div class="flex-1 flex flex-col h-full overflow-hidden bg-bg relative">
    <!-- Messages Viewport -->
    <div class="flex-1 overflow-y-auto px-5 py-6 space-y-6 custom-scrollbar" bind:this={messagesViewportRef}>
      {#if !activeConv || activeConv.messages.length === 0}
        <!-- Welcome Hero (ChatGPT / Oprel style) -->
        <div class="flex flex-col items-center justify-center h-full min-h-[380px] text-center max-w-lg mx-auto space-y-4">
          <img
            src="/dara.png"
            alt="Dara"
            class="w-14 h-14 object-contain mb-1"
          />
          <div>
            <h2 class="text-xl font-bold tracking-tight text-text">What can I help with today?</h2>
            <p class="text-xs text-text-muted mt-1">
              Ask anything using <span class="font-bold text-text">{loadedModelInfo?.dara.name ?? "local models"}</span> running entirely offline on your GPU.
            </p>
          </div>

          <div class="flex flex-wrap gap-2 justify-center pt-2">
            {#each ["Explain quantum entanglement", "Write a fast Python merge sort", "Analyze this image for objects", "Optimize SQL query"] as prompt}
              <button
                type="button"
                onclick={() => {
                  inputText = prompt;
                  handleSend();
                }}
                class="px-3 py-1.5 rounded-lg border border-border bg-surface hover:bg-surface-raised text-xs text-text-muted hover:text-text transition-colors cursor-pointer"
              >
                {prompt}
              </button>
            {/each}
          </div>
        </div>
      {:else}
        <!-- Message Bubbles Container -->
        <div class="max-w-3xl mx-auto space-y-6">
          {#each activeConv.messages as msg, idx (msg.id)}
            {#if msg.role === "user"}
              <!-- User Message Bubble (ChatGPT / Claude / assistant-ui style) -->
              <div class="flex flex-col items-end group">
                {#if isEditingId === msg.id}
                  <!-- Claude/ChatGPT Edit Card -->
                  <div class="w-full max-w-2xl rounded-2xl bg-surface border border-primary/40 shadow-md p-3.5 space-y-3">
                    <textarea
                      bind:value={editText}
                      rows={3}
                      class="w-full bg-transparent p-0 text-sm text-text placeholder:text-text-subtle resize-none border-0 outline-none ring-0 leading-relaxed"
                      style="outline: none !important; box-shadow: none !important; border: none !important;"
                      placeholder="Edit your message..."
                      onkeydown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey) {
                          e.preventDefault();
                          submitEdit(msg.id);
                        } else if (e.key === "Escape") {
                          isEditingId = null;
                        }
                      }}
                    ></textarea>
                    <div class="flex items-center justify-end gap-2 pt-2 border-t border-border/40">
                      <button
                        type="button"
                        onclick={() => (isEditingId = null)}
                        class="px-3 py-1.5 rounded-lg border border-border bg-surface hover:bg-surface-raised text-xs font-semibold text-text-muted hover:text-text transition-colors cursor-pointer"
                      >
                        Cancel
                      </button>
                      <button
                        type="button"
                        onclick={() => submitEdit(msg.id)}
                        class="px-3.5 py-1.5 rounded-lg bg-primary hover:bg-primary-hover text-white text-xs font-semibold transition-all cursor-pointer shadow-sm"
                      >
                        Save & Resend
                      </button>
                    </div>
                  </div>
                {:else}
                  <!-- Message Bubble -->
                  <div class="max-w-[75%] rounded-2xl rounded-tr-sm px-4 py-2.5 bg-primary/10 border border-primary/20 text-sm text-text space-y-2 shadow-xs">
                    {#if msg.images && msg.images.length > 0}
                      <div class="flex flex-wrap gap-2">
                        {#each msg.images as img, i (i)}
                          <img
                            src={img}
                            alt="Attachment"
                            class="max-w-xs max-h-48 rounded-lg border border-border object-cover"
                          />
                        {/each}
                      </div>
                    {/if}
                    <p class="leading-relaxed whitespace-pre-wrap break-words">{msg.content}</p>
                  </div>

                  <!-- Hover Action Icons (ChatGPT/Claude style: edit pencil & copy) -->
                  <div class="opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity flex items-center gap-1 mt-1 pr-1">
                    <button
                      type="button"
                      aria-label="Edit message"
                      onclick={() => startEdit(msg)}
                      class="p-1 rounded-md text-text-subtle hover:text-text hover:bg-surface-raised transition-colors cursor-pointer"
                      title="Edit message"
                    >
                      <Icon name="pencil" size={13} />
                    </button>
                    <button
                      type="button"
                      aria-label="Copy message"
                      onclick={() => copyMessage(msg.content, msg.id)}
                      class="p-1 rounded-md text-text-subtle hover:text-text hover:bg-surface-raised transition-colors cursor-pointer"
                      title={copiedMessageId === msg.id ? "Copied!" : "Copy message"}
                    >
                      {#if copiedMessageId === msg.id}
                        <Icon name="check" size={13} class="text-success" />
                      {:else}
                        <Icon name="copy" size={13} />
                      {/if}
                    </button>
                  </div>
                {/if}
              </div>
            {:else if msg.role === "assistant"}
              {@const prevMsg = activeConv.messages[idx - 1]}
              {@const isDuplicate = prevMsg && prevMsg.role === "assistant" && prevMsg.content === msg.content}
              {#if !isDuplicate}
                {@const isStreamingThis = chatStore.isGenerating && idx === activeConv.messages.length - 1}
                <!-- Assistant Message -->
                <div class="flex gap-3 group">
                  <!-- Dara Avatar using /dara.png without round border -->
                  <img
                    src="/dara.png"
                    alt="Dara AI"
                    class="w-7 h-7 object-contain shrink-0 mt-0.5 {isStreamingThis ? 'animate-pulse' : ''}"
                  />

                  <div class="flex-1 min-w-0 space-y-1">
                    <!-- Markdown Renderer -->
                    <div class="text-sm leading-relaxed text-text">
                      <Markdown content={msg.content} />
                    </div>

                    <!-- Telemetry & Action Buttons (⚡ tok/s badge, tokens count, copy, thumbs, regenerate) -->
                    {#if !isStreamingThis && msg.content}
                      <div class="pt-1">
                        <MessageActions
                          content={msg.content}
                          tps={msg.stats?.decodeTokensPerSecond}
                          tokCount={msg.stats?.totalTokens ?? msg.stats?.decodeTokens}
                          dur={msg.stats?.totalMs ?? ((msg.stats?.decodeMs ?? 0) + (msg.stats?.prefillMs ?? 0))}
                          onregenerate={() => chatStore.regenerate(msg.id, serverStore.loadedModel ?? undefined)}
                        />
                      </div>
                    {/if}
                  </div>
                </div>
              {/if}
            {/if}
          {/each}
        </div>
      {/if}
    </div>

    <!-- 3. Bottom: Unified Composer Container (assistant-ui/elements/composer style) -->
    <div class="p-4 bg-bg shrink-0 max-w-3xl w-full mx-auto">
      <div class="relative rounded-2xl bg-surface border border-border focus-within:border-primary focus-within:ring-2 focus-within:ring-primary/20 transition-all shadow-sm p-3 space-y-2">
        <!-- Attachments Tray Inside Composer -->
        {#if attachedImages.length > 0}
          <div class="flex flex-wrap gap-2 pb-2 border-b border-border/40">
            {#each attachedImages as img, i (i)}
              <div class="relative group/thumb">
                <img src={img} alt="Attachment thumbnail" class="w-14 h-14 object-cover rounded-xl border border-border shadow-xs" />
                <button
                  type="button"
                  aria-label="Remove image"
                  onclick={() => {
                    attachedImages = attachedImages.filter((_, idx) => idx !== i);
                  }}
                  class="absolute -top-1.5 -right-1.5 bg-text text-bg hover:bg-danger hover:text-white rounded-full w-4 h-4 flex items-center justify-center text-[10px] cursor-pointer shadow-sm transition-colors"
                >
                  ✕
                </button>
              </div>
            {/each}
          </div>
        {/if}

        <!-- Auto-Expanding Textarea (Zero inner border/outline) -->
        <textarea
          bind:this={textareaRef}
          bind:value={inputText}
          oninput={autoResizeTextarea}
          onkeydown={handleKeyDown}
          placeholder="Send a message (Shift+Enter for newline)..."
          rows={1}
          class="w-full bg-transparent p-0 text-sm text-text placeholder:text-text-subtle resize-none border-0 outline-none ring-0 focus:outline-none focus:ring-0 leading-relaxed min-h-[44px] max-h-[160px]"
          style="outline: none !important; box-shadow: none !important; border: none !important;"
        ></textarea>

        <!-- Bottom Toolbar Inside Prompt Container -->
        <div class="flex items-center justify-between pt-2.5 mt-1 border-t border-border/40">
          <div class="flex items-center gap-2">
            <!-- Hidden file input -->
            <input
              type="file"
              accept="image/png,image/jpeg,image/webp"
              bind:this={fileInputRef}
              onchange={handleFileSelect}
              class="hidden"
            />

            <!-- + Attach Image / Tool Button -->
            <button
              type="button"
              onclick={() => fileInputRef?.click()}
              class="p-1.5 rounded-lg text-text-muted hover:text-text hover:bg-surface-raised transition-colors cursor-pointer"
              title={isVisionSupported ? "Upload image for analysis" : "Image upload"}
              aria-label="Attach image"
            >
              <Icon name="plus" size={16} />
            </button>

            <!-- Generation Settings Popover -->
            <Popover side="top">
              {#snippet trigger({ isOpen, toggle })}
                <button
                  type="button"
                  onclick={toggle}
                  aria-expanded={isOpen}
                  class="p-1.5 rounded-lg text-text-muted hover:text-text hover:bg-surface-raised transition-colors cursor-pointer"
                  title="Generation parameters"
                  aria-label="Generation parameters"
                >
                  <Icon name="sliders-horizontal" size={15} />
                </button>
              {/snippet}
              {#snippet children({ close })}
                <div class="space-y-4 w-68 text-left p-1">
                  <h4 class="text-xs font-bold text-text uppercase tracking-wider">Generation Controls</h4>
                  {#if activeConv}
                    <Toggle label="Flash Speculative Drafter" bind:checked={activeConv.flash} />
                    <Slider label="Temperature" min={0.1} max={2.0} step={0.1} bind:value={activeConv.temperature} />
                    <Slider label="Max Tokens" min={128} max={4096} step={128} unit=" tok" bind:value={activeConv.maxTokens} />
                    <Textarea label="System Prompt" rows={2} bind:value={activeConv.systemPrompt} />
                  {/if}
                  <Button size="sm" variant="primary" class="w-full" onclick={close}>Done</Button>
                </div>
              {/snippet}
            </Popover>

            <span class="text-border">|</span>

            <!-- assistant-ui Context Window Meter Pill -->
            <div class="relative">
              <button
                type="button"
                onclick={() => (isContextPopoverOpen = !isContextPopoverOpen)}
                class="h-6 px-2.5 rounded-full border border-border bg-surface-raised hover:bg-surface-sunken transition-all flex items-center gap-1.5 text-[11px] font-mono select-none cursor-pointer"
                aria-label="Context utilization"
              >
                <span
                  class="w-1.5 h-1.5 rounded-full"
                  style="background-color: {contextPercent >= 85 ? '#f43f5e' : contextPercent >= 60 ? '#f59e0b' : '#22c55e'};"
                ></span>
                <span>
                  <NumberTicker value={activeConv?.context ? activeConv.context.used : 0} /> /{" "}
                  {activeConv?.context ? Math.round(activeConv.context.capacity / 1024) + "k" : "128k"}
                </span>
                <span class="text-[9px] text-text-subtle">({contextPercent}%)</span>
                <Icon name="chevron-down" size={10} class="text-text-subtle" />
              </button>

              {#if isContextPopoverOpen}
                <!-- Backdrop -->
                <!-- svelte-ignore a11y_no_static_element_interactions -->
                <!-- svelte-ignore a11y_click_events_have_key_events -->
                <div class="fixed inset-0 z-40" onclick={() => (isContextPopoverOpen = false)}></div>

                <!-- assistant-ui Context Breakdown Element -->
                <div class="absolute bottom-full left-0 mb-2 z-50 animate-in fade-in zoom-in-95 duration-150">
                  <ContextBreakdown
                    used={activeConv?.context ? activeConv.context.used : 0}
                    capacity={activeConv?.context ? activeConv.context.capacity : 131072}
                    systemTokens={activeConv?.systemPrompt ? Math.round(activeConv.systemPrompt.length / 4) : 0}
                    historyTokens={activeConv?.context
                      ? Math.max(
                          0,
                          activeConv.context.used -
                            (activeConv?.systemPrompt ? Math.round(activeConv.systemPrompt.length / 4) : 0)
                        )
                      : 0}
                  />
                </div>
              {/if}
            </div>
          </div>

          <!-- Send / Stop Button -->
          <div class="flex items-center gap-2">
            {#if chatStore.isGenerating}
              <button
                type="button"
                onclick={() => chatStore.stopGeneration()}
                class="flex items-center gap-1.5 h-8 px-3 rounded-xl bg-danger hover:bg-danger-hover text-white text-xs font-semibold transition-all cursor-pointer shadow-sm"
              >
                <Icon name="square" size={12} />
                <span>Stop</span>
              </button>
            {:else}
              <button
                type="button"
                disabled={!inputText.trim() && attachedImages.length === 0}
                onclick={handleSend}
                class="flex items-center gap-1.5 h-8 px-3.5 rounded-xl bg-primary hover:bg-primary-hover disabled:opacity-35 disabled:cursor-not-allowed text-white text-xs font-semibold transition-all cursor-pointer shadow-sm"
              >
                <span>Send</span>
                <Icon name="send" size={12} />
              </button>
            {/if}
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
