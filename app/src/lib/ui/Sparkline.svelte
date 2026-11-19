<script lang="ts">
  interface Props {
    data: number[];
    width?: number;
    height?: number;
    label?: string;
    class?: string;
  }

  const {
    data = [],
    width = 120,
    height = 28,
    label = "Metric sparkline",
    class: className = ""
  }: Props = $props();

  const min = $derived(data.length > 0 ? Math.min(...data) : 0);
  const max = $derived(data.length > 0 ? Math.max(...data, min + 1) : 1);

  const points = $derived.by(() => {
    if (data.length < 2) return "";
    const step = width / (data.length - 1);
    return data
      .map((val, i) => {
        const x = i * step;
        const y = height - ((val - min) / (max - min)) * (height - 4) - 2;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  });

  const lastValue = $derived(data[data.length - 1] ?? 0);
</script>

<div class="inline-flex items-center space-x-2 {className}" aria-label={label}>
  <svg {width} {height} viewBox="0 0 {width} {height}" class="overflow-visible" aria-hidden="true">
    {#if points}
      <polyline
        fill="none"
        stroke="var(--color-primary)"
        stroke-width="1.5"
        stroke-linecap="round"
        stroke-linejoin="round"
        {points}
      />
    {/if}
  </svg>

  <span class="text-xs font-mono tabular-nums text-text-muted">
    {lastValue.toFixed(1)}
  </span>
</div>
