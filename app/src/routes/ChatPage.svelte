<script lang="ts">
  import { chatStore, type UiMessage } from "../lib/stores/chat.svelte.ts";
  import { serverStore } from "../lib/stores/server.svelte.ts";
  import { modelsStore } from "../lib/stores/models.svelte.ts";
  import Button from "../lib/ui/Button.svelte";
  import IconButton from "../lib/ui/IconButton.svelte";
  import Badge from "../lib/ui/Badge.svelte";
  import Meter from "../lib/ui/Meter.svelte";
  import Popover from "../lib/ui/Popover.svelte";
  import Toggle from "../lib/ui/Toggle.svelte";
  import Slider from "../lib/ui/Slider.svelte";
  import Textarea from "../lib/ui/Textarea.svelte";
  import Markdown from "../lib/ui/Markdown.svelte";
  import EmptyState from "../lib/ui/EmptyState.svelte";
  import { toast } from "../lib/ui/Toast.svelte";

  let inputText = $state("");
  let attachedImages = $state<string[]>([]);
  let fileInputRef: HTMLInputElement | null = $state(null);
  let isEditingId = $state<string | null>(null);
  let editText = $state("");

  const activeConv = $derived(chatStore.activeConversation);
  const loadedModelInfo = $derived(modelsStore.loadedModel);
  const isVisionSupported = $derived(loadedModelInfo?.dara.vision.active ?? false);

  function handleSend() {
    if ((!inputText.trim() && attachedImages.length === 0) || chatStore.isGenerating) return;
    const text = inputText;
    const imgs = [...attachedImages];
    inputText = "";
    attachedImages = [];
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

  function copyMessage(text: string) {
    void navigator.clipboard.writeText(text);
    toast.show("Message copied to clipboard", "info");
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
</script>

<div class="flex-1 flex h-full overflow-hidden text-left bg-bg">
  <!-- 1. Left: Conversations Rail -->
  <aside
    class="w-60 border-r border-border bg-bg-subtle flex flex-col justify-between shrink-0 select-none overflow-hidden"
    aria-label="Conversations"
  >
    <div class="p-3 space-y-3 flex-1 flex flex-col overflow-hidden">
      <!-- New Chat Button -->
      <Button
        variant="primary"
        size="sm"
        class="w-full justify-start gap-2 font-medium"
        onclick={() => chatStore.createConversation()}
      >
        <svg
          class="w-4 h-4"
          viewBox="0 0 20 20"
          fill="none"
          stroke="currentColor"
          stroke-width="1.75"
          aria-hidden="true"
        >
          <path d="M10 4v12M4 10h12" stroke-linecap="round" />
        </svg>
        <span>New Chat</span>
      </Button>

      <!-- Thread List -->
      <div class="flex-1 overflow-y-auto space-y-1">
        {#each chatStore.conversations as conv (conv.id)}
          {@const isActive = chatStore.activeConversationId === conv.id}
          <div
            role="button"
            tabindex="0"
            onclick={() => chatStore.selectConversation(conv.id)}
            onkeydown={(e) => {
              if (e.key === "Enter" || e.key === " ") chatStore.selectConversation(conv.id);
            }}
            class="group flex items-center justify-between px-2.5 py-2 rounded-md text-xs cursor-pointer transition-colors {isActive
              ? 'bg-primary-subtle text-on-primary-subtle font-semibold'
              : 'text-text-muted hover:text-text hover:bg-surface-raised'}"
          >
            <span class="truncate flex-1">{conv.title}</span>
            <button
              type="button"
              aria-label="Delete conversation"
              onclick={(e) => {
                e.stopPropagation();
                chatStore.deleteConversation(conv.id);
              }}
              class="opacity-0 group-hover:opacity-100 p-0.5 rounded text-text-subtle hover:text-danger cursor-pointer"
            >
              <svg
                class="w-3.5 h-3.5"
                viewBox="0 0 16 16"
                fill="none"
                stroke="currentColor"
                stroke-width="1.5"
                aria-hidden="true"
              >
                <path
                  d="M3 4h10M5 4V3a1 1 0 011-1h4a1 1 0 011 1v1M6 7v5M10 7v5"
                  stroke-linecap="round"
                />
              </svg>
            </button>
          </div>
        {/each}
      </div>
    </div>
  </aside>

  <!-- 2. Center: Chat Thread & Composer -->
  <div class="flex-1 flex flex-col h-full overflow-hidden">
    <!-- Messages Scroll Area -->
    <div
      class="flex-1 overflow-y-auto p-6 space-y-6 max-w-3xl w-full mx-auto"
      role="log"
      aria-live="polite"
    >
      {#if !activeConv || activeConv.messages.length === 0}
        <EmptyState
          title="How can Dara help today?"
          description="Local private inference running directly on your Windows GPU. No telemetry, no cloud logging."
        >
          {#snippet actions()}
            <div class="flex flex-wrap gap-2 justify-center">
              <Button
                size="sm"
                variant="secondary"
                onclick={() => {
                  inputText = "Explain quantization in plain words";
                  handleSend();
                }}
              >
                Explain quantization
              </Button>
              <Button
                size="sm"
                variant="secondary"
                onclick={() => {
                  inputText = "Write a fast Python merge sort function";
                  handleSend();
                }}
              >
                Python merge sort
              </Button>
            </div>
          {/snippet}
        </EmptyState>
      {:else}
        {#each activeConv.messages as msg (msg.id)}
          {#if msg.role === "user"}
            <!-- User message bubble: right-aligned, primary-subtle -->
            <div class="flex justify-end">
              <div
                class="max-w-[85%] rounded-2xl rounded-tr-sm bg-primary-subtle text-on-primary-subtle px-4 py-3 space-y-2 border border-primary-subtle"
              >
                {#if msg.images && msg.images.length > 0}
                  <div class="flex flex-wrap gap-2">
                    {#each msg.images as imgUrl, i (i)}
                      <img
                        src={imgUrl}
                        alt="User attachment"
                        class="w-24 h-24 object-cover rounded-lg border border-border"
                      />
                    {/each}
                  </div>
                {/if}

                {#if isEditingId === msg.id}
                  <div class="space-y-2">
                    <Textarea bind:value={editText} rows={2} />
                    <div class="flex justify-end space-x-2">
                      <Button size="sm" variant="ghost" onclick={() => (isEditingId = null)}
                        >Cancel</Button
                      >
                      <Button size="sm" variant="primary" onclick={() => submitEdit(msg.id)}
                        >Resend</Button
                      >
                    </div>
                  </div>
                {:else}
                  <p class="text-sm whitespace-pre-wrap leading-relaxed">{msg.content}</p>

                  <div
                    class="flex items-center justify-end space-x-2 pt-1 opacity-70 hover:opacity-100 text-xs"
                  >
                    <button
                      type="button"
                      aria-label="Copy message"
                      onclick={() => copyMessage(msg.content)}
                      class="hover:underline cursor-pointer"
                    >
                      Copy
                    </button>
                    <span>•</span>
                    <button
                      type="button"
                      aria-label="Edit message"
                      onclick={() => startEdit(msg)}
                      class="hover:underline cursor-pointer"
                    >
                      Edit
                    </button>
                  </div>
                {/if}
              </div>
            </div>
          {:else if msg.role === "assistant"}
            <!-- Assistant message: plain text on page with Dara geometric mark -->
            <div class="flex items-start space-x-3.5 group">
              <!-- Geometric mark -->
              <div
                class="w-7 h-7 rounded-lg bg-primary-subtle flex items-center justify-center text-primary shrink-0 mt-0.5 select-none"
              >
                <svg
                  class="w-4 h-4"
                  viewBox="0 0 64 64"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                  aria-hidden="true"
                >
                  <rect x="8" y="8" width="18" height="48" rx="6" fill="currentColor" />
                  <path
                    d="M26 8H38C49.0457 8 58 16.9543 58 28V36C58 47.0457 49.0457 56 38 56H26V8Z"
                    fill="currentColor"
                    fill-opacity="0.85"
                  />
                </svg>
              </div>

              <div class="flex-1 space-y-2 overflow-hidden">
                <div class="flex items-center space-x-2 text-xs">
                  <span class="font-semibold text-text">Dara</span>
                  {#if msg.interrupted}
                    <Badge variant="warning" size="sm">interrupted</Badge>
                  {/if}
                </div>

                <!-- Message Content -->
                <Markdown content={msg.content} />

                <!-- Streaming indicator -->
                {#if chatStore.isGenerating && msg.id === activeConv.messages[activeConv.messages.length - 1]?.id}
                  <div class="flex items-center space-x-2 pt-1" aria-label="Generating response">
                    <span class="w-2 h-2 rounded-full bg-primary animate-pulse"></span>
                    <span class="text-xs text-text-subtle font-mono">Thinking...</span>
                  </div>
                {/if}

                <!-- Per-message stats & actions footer -->
                <div
                  class="flex items-center justify-between text-xs text-text-subtle pt-1 border-t border-border border-dashed"
                >
                  <div class="flex items-center space-x-3 font-mono text-[11px] tabular-nums">
                    {#if msg.stats?.decodeTokensPerSecond}
                      <span>{msg.stats.decodeTokensPerSecond.toFixed(1)} tok/s</span>
                    {/if}
                    {#if msg.stats?.ttftMs}
                      <span>TTFT: {msg.stats.ttftMs.toFixed(0)}ms</span>
                    {/if}
                    {#if msg.stats?.cachedTokens}
                      <span>Cached: {msg.stats.cachedTokens}</span>
                    {/if}
                    {#if msg.stats?.speculative}
                      <span class="text-primary font-medium"
                        >Flash: {msg.stats.speculative.acceptance.toFixed(0)}%</span
                      >
                    {/if}
                  </div>

                  <div
                    class="flex items-center space-x-2 opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <button
                      type="button"
                      aria-label="Copy reply"
                      onclick={() => copyMessage(msg.content)}
                      class="hover:text-text cursor-pointer"
                    >
                      Copy
                    </button>
                    <span>•</span>
                    <button
                      type="button"
                      aria-label="Regenerate reply"
                      onclick={() => chatStore.regenerate(serverStore.loadedModel ?? undefined)}
                      class="hover:text-text cursor-pointer"
                    >
                      Regenerate
                    </button>
                  </div>
                </div>
              </div>
            </div>
          {/if}
        {/each}
      {/if}
    </div>

    <!-- 3. Bottom: Context Meter & Composer Area -->
    <div class="p-4 border-t border-border bg-surface shrink-0 max-w-3xl w-full mx-auto space-y-3">
      <!-- Context Meter & Compaction Note -->
      {#if activeConv?.context}
        <div class="space-y-1">
          <Meter
            label="Context Length"
            used={activeConv.context.used}
            capacity={activeConv.context.capacity}
            warnThreshold={0.85}
          />
          {#if activeConv.context.compacted}
            <p class="text-[11px] text-text-muted italic">
              Older messages were summarized to fit ({activeConv.context.summarizedMessages} summarized,
              {activeConv.context.droppedMessages} dropped).
            </p>
          {/if}
        </div>
      {/if}

      <!-- Image attachments previews -->
      {#if attachedImages.length > 0}
        <div class="flex flex-wrap gap-2 p-2 bg-surface-sunken rounded-lg border border-border">
          {#each attachedImages as img, i (i)}
            <div class="relative group">
              <img
                src={img}
                alt="Attachment thumbnail"
                class="w-14 h-14 object-cover rounded-md border border-border"
              />
              <button
                type="button"
                aria-label="Remove image"
                onclick={() => {
                  attachedImages = attachedImages.filter((_, idx) => idx !== i);
                }}
                class="absolute -top-1.5 -right-1.5 bg-danger text-on-primary rounded-full w-4 h-4 flex items-center justify-center text-[10px] cursor-pointer"
              >
                ✕
              </button>
            </div>
          {/each}
        </div>
      {/if}

      <!-- Composer Card -->
      <div
        class="rounded-xl border border-border bg-surface-sunken p-2.5 space-y-2 focus-within:border-primary transition-colors"
      >
        <textarea
          bind:value={inputText}
          onkeydown={handleKeyDown}
          placeholder="Send a message (Shift+Enter for newline)..."
          rows={2}
          class="w-full bg-transparent p-1 text-sm text-text placeholder:text-text-subtle resize-none outline-none leading-relaxed"
        ></textarea>

        <div class="flex items-center justify-between pt-1 border-t border-border">
          <!-- Attach Image & Settings Popover -->
          <div class="flex items-center space-x-1">
            <input
              type="file"
              accept="image/png,image/jpeg,image/webp"
              bind:this={fileInputRef}
              onchange={handleFileSelect}
              class="hidden"
            />

            <IconButton
              aria-label={isVisionSupported
                ? "Attach image"
                : "This model was loaded without image support"}
              disabled={!isVisionSupported}
              onclick={() => fileInputRef?.click()}
            >
              <svg
                class="w-4 h-4 text-text-muted"
                viewBox="0 0 20 20"
                fill="none"
                stroke="currentColor"
                stroke-width="1.5"
                aria-hidden="true"
              >
                <path
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L18 14m-5-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                />
              </svg>
            </IconButton>

            <!-- Generation Settings Popover -->
            <Popover>
              {#snippet trigger({ isOpen, toggle })}
                <IconButton
                  aria-label="Generation parameters"
                  onclick={toggle}
                  aria-expanded={isOpen}
                >
                  <svg
                    class="w-4 h-4 text-text-muted"
                    viewBox="0 0 20 20"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="1.5"
                    aria-hidden="true"
                  >
                    <path d="M5 4h10M3 10h14M7 16h6" stroke-linecap="round" />
                  </svg>
                </IconButton>
              {/snippet}
              {#snippet children({ close })}
                <div class="space-y-4 w-64 text-left">
                  <h4 class="text-xs font-bold text-text uppercase tracking-wider">
                    Generation Controls
                  </h4>
                  {#if activeConv}
                    <Toggle label="Flash Speculative Drafter" bind:checked={activeConv.flash} />
                    <Slider
                      label="Temperature"
                      min={0.1}
                      max={2.0}
                      step={0.1}
                      bind:value={activeConv.temperature}
                    />
                    <Slider
                      label="Max Tokens"
                      min={128}
                      max={4096}
                      step={128}
                      unit=" tok"
                      bind:value={activeConv.maxTokens}
                    />
                    <Textarea label="System Prompt" rows={2} bind:value={activeConv.systemPrompt} />
                  {/if}
                  <Button size="sm" variant="primary" class="w-full" onclick={close}>Done</Button>
                </div>
              {/snippet}
            </Popover>
          </div>

          <!-- Send / Stop Button -->
          <div class="flex items-center space-x-2">
            {#if chatStore.isGenerating}
              <Button variant="danger" size="sm" onclick={() => chatStore.stopGeneration()}>
                <span class="w-2 h-2 bg-on-primary rounded-xs mr-1.5"></span>
                Stop
              </Button>
            {:else}
              <Button
                variant="primary"
                size="sm"
                disabled={!inputText.trim() && attachedImages.length === 0}
                onclick={handleSend}
              >
                Send
              </Button>
            {/if}
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
