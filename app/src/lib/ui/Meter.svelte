<script lang="ts">
  interface Props {
    label: string;
    used: number;
    capacity: number;
    unit?: string;
    warnThreshold?: number; // 0 to 1, default 0.85
    class?: string;
  }

  const {
    label,
    used,
    capacity,
    unit = "tokens",
    warnThreshold = 0.85,
    class: className = ""
  }: Props = $props();

  const ratio = $derived(capacity > 0 ? used / capacity : 0);
  const percent = $derived(Math.min(100, Math.max(0, Math.round(ratio * 100))));
  const isWarning = $derived(ratio >= warnThreshold);

  function formatNumber(num: number): string {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(0)}K`;
    return num.toLocaleString();
  }
</script>

<div class="flex flex-col space-y-1.5 w-full text-left {className}">
  <div class="flex items-center justify-between text-xs">
    <span class="font-medium text-text-muted">{label}</span>
    <span
      class="font-mono tabular-nums {isWarning ? 'text-warning font-semibold' : 'text-text-muted'}"
    >
      {formatNumber(used)} / {formatNumber(capacity)}
      {unit} ({percent}%)
    </span>
  </div>

  <div
    role="meter"
    aria-valuenow={used}
    aria-valuemin={0}
    aria-valuemax={capacity}
    aria-label={label}
    class="w-full h-1.5 rounded-full overflow-hidden bg-surface-sunken border border-border"
  >
    <div
      class="h-full rounded-full transition-all duration-base {isWarning
        ? 'bg-warning'
        : 'bg-primary'}"
      style="width: {percent}%;"
    ></div>
  </div>
</div>
