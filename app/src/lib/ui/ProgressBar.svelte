<script lang="ts">
  interface Props {
    value?: number;
    max?: number;
    label?: string;
    showPercent?: boolean;
    class?: string;
  }

  const { value, max = 100, label, showPercent = false, class: className = "" }: Props = $props();

  const isIndeterminate = $derived(value === undefined);
  const percent = $derived(
    isIndeterminate ? 0 : Math.min(100, Math.max(0, Math.round(((value ?? 0) / max) * 100)))
  );
</script>

<div class="flex flex-col space-y-1.5 w-full text-left {className}">
  {#if label || showPercent}
    <div class="flex items-center justify-between text-xs">
      {#if label}
        <span class="font-medium text-text">{label}</span>
      {/if}
      {#if showPercent && !isIndeterminate}
        <span class="font-mono text-text-muted">{percent}%</span>
      {/if}
    </div>
  {/if}

  <div
    role="progressbar"
    aria-valuenow={isIndeterminate ? undefined : value}
    aria-valuemin={isIndeterminate ? undefined : 0}
    aria-valuemax={isIndeterminate ? undefined : max}
    aria-label={label ?? "Progress"}
    class="w-full h-2 rounded-full overflow-hidden bg-surface-sunken border border-border"
  >
    {#if isIndeterminate}
      <div class="h-full w-1/3 bg-primary rounded-full animate-pulse"></div>
    {:else}
      <div
        class="h-full bg-primary rounded-full transition-all duration-base"
        style="width: {percent}%;"
      ></div>
    {/if}
  </div>
</div>
