<script lang="ts">
  import Icon from "./Icon.svelte";

  interface Props {
    used: number;
    capacity: number;
    systemTokens?: number;
    historyTokens?: number;
    completionTokens?: number;
    class?: string;
  }

  const {
    used,
    capacity,
    systemTokens = 0,
    historyTokens = 0,
    completionTokens = 0,
    class: className = ""
  }: Props = $props();

  const totalCapacity = $derived(capacity > 0 ? capacity : 131072);
  const usedPercent = $derived(Math.min(100, Math.round((used / totalCapacity) * 100)));

  const systemPct = $derived(Math.min(100, (systemTokens / totalCapacity) * 100));
  const historyPct = $derived(Math.min(100, (historyTokens / totalCapacity) * 100));
  const completionPct = $derived(Math.min(100, (completionTokens / totalCapacity) * 100));
</script>

<div class="rounded-2xl border border-border bg-surface-raised p-4 text-left shadow-xl space-y-4 select-none w-80 {className}">
  <!-- Header -->
  <div class="flex items-center justify-between">
    <div class="flex items-center gap-1.5 text-xs font-bold text-text uppercase tracking-wider">
      <Icon name="database" size={13} class="text-primary" />
      <span>Context Breakdown</span>
    </div>
    <span class="text-xs font-mono font-bold text-text">
      {used.toLocaleString()} / {Math.round(totalCapacity / 1024)}k ({usedPercent}%)
    </span>
  </div>

  <!-- Multi-Segment Progress Bar -->
  <div class="h-2.5 w-full bg-surface-sunken rounded-full overflow-hidden flex border border-border/40">
    {#if systemPct > 0}
      <div
        class="h-full bg-indigo-500 transition-all duration-300"
        style="width: {Math.max(systemPct, 1)}%;"
        title="System Prompt: {systemTokens} tokens"
      ></div>
    {/if}
    {#if historyPct > 0}
      <div
        class="h-full bg-primary transition-all duration-300"
        style="width: {Math.max(historyPct, 1)}%;"
        title="Chat History: {historyTokens} tokens"
      ></div>
    {/if}
    {#if completionPct > 0}
      <div
        class="h-full bg-emerald-500 transition-all duration-300"
        style="width: {Math.max(completionPct, 1)}%;"
        title="Completion: {completionTokens} tokens"
      ></div>
    {/if}
    {#if usedPercent === 0}
      <div class="h-full bg-primary/20 w-1 rounded-full"></div>
    {/if}
  </div>

  <!-- Legend / Token Allocation -->
  <div class="space-y-2 pt-1 text-xs">
    <div class="flex items-center justify-between text-text-subtle">
      <div class="flex items-center gap-2">
        <span class="size-2 rounded-full bg-indigo-500"></span>
        <span>System Prompt</span>
      </div>
      <span class="font-mono font-medium text-text">{systemTokens.toLocaleString()} tok</span>
    </div>

    <div class="flex items-center justify-between text-text-subtle">
      <div class="flex items-center gap-2">
        <span class="size-2 rounded-full bg-primary"></span>
        <span>Conversation History</span>
      </div>
      <span class="font-mono font-medium text-text">{historyTokens.toLocaleString()} tok</span>
    </div>

    {#if completionTokens > 0}
      <div class="flex items-center justify-between text-text-subtle">
        <div class="flex items-center gap-2">
          <span class="size-2 rounded-full bg-emerald-500"></span>
          <span>Recent Response</span>
        </div>
        <span class="font-mono font-medium text-text">{completionTokens.toLocaleString()} tok</span>
      </div>
    {/if}

    <div class="flex items-center justify-between text-text-subtle border-t border-border/50 pt-2">
      <div class="flex items-center gap-2">
        <span class="size-2 rounded-full bg-surface-sunken border border-border"></span>
        <span>Remaining Free Buffer</span>
      </div>
      <span class="font-mono font-medium text-text-muted">
        {Math.max(0, totalCapacity - used).toLocaleString()} tok
      </span>
    </div>
  </div>
</div>
