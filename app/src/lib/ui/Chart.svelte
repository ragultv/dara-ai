<script lang="ts">
  interface Props {
    label: string;
    value: string | number;
    delta?: string;
    points: number[];
    visibleCount?: number;
    variant?: "area" | "line" | "bar";
    class?: string;
  }

  const {
    label,
    value,
    delta,
    points = [],
    visibleCount,
    variant = "area",
    class: className = ""
  }: Props = $props();

  const activePoints = $derived.by(() => {
    if (!points || points.length === 0) return [];
    const count = visibleCount !== undefined ? Math.min(visibleCount, points.length) : points.length;
    return points.slice(0, count);
  });

  const chartGeometry = $derived.by(() => {
    const pts = activePoints;
    if (pts.length < 2) return { path: "", area: "", min: 0, max: 1 };

    const min = Math.min(...pts);
    const max = Math.max(...pts);
    const range = max - min || 1;
    const width = 240;
    const height = 48;
    const padding = 4;

    const coords = pts.map((p, idx) => {
      const x = padding + (idx / (pts.length - 1)) * (width - padding * 2);
      const y = height - padding - ((p - min) / range) * (height - padding * 2);
      return { x, y };
    });

    const linePath = coords.reduce((acc, pt, idx) => {
      return idx === 0 ? `M ${pt.x} ${pt.y}` : `${acc} L ${pt.x} ${pt.y}`;
    }, "");

    const lastX = coords[coords.length - 1]?.x ?? width;
    const firstX = coords[0]?.x ?? 0;
    const areaPath = `${linePath} L ${lastX} ${height} L ${firstX} ${height} Z`;

    return { path: linePath, area: areaPath, min, max };
  });

  const isPositiveDelta = $derived(delta ? !delta.startsWith("-") : true);
</script>

<div class="rounded-xl border border-border bg-surface p-4 text-left shadow-sm space-y-2 select-none {className}">
  <!-- Header: Label & Delta Pill -->
  <div class="flex items-center justify-between text-xs">
    <span class="font-medium text-text-subtle truncate">{label}</span>
    {#if delta}
      <span
        class="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono font-semibold {isPositiveDelta
          ? 'bg-success/15 text-success border border-success/20'
          : 'bg-danger/15 text-danger border border-danger/20'}"
      >
        {delta}
      </span>
    {/if}
  </div>

  <!-- Value -->
  <div class="text-xl font-bold font-mono text-text tracking-tight">
    {value}
  </div>

  <!-- SVG Chart -->
  {#if activePoints.length >= 2}
    <div class="w-full h-12 pt-1 overflow-hidden">
      <svg viewBox="0 0 240 48" class="w-full h-full overflow-visible" preserveAspectRatio="none">
        <defs>
          <linearGradient id="aui-chart-grad-{label.replace(/\s+/g, '-')}" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="var(--color-primary)" stop-opacity="0.35" />
            <stop offset="100%" stop-color="var(--color-primary)" stop-opacity="0.0" />
          </linearGradient>
        </defs>

        {#if variant === "area"}
          <path
            d={chartGeometry.area}
            fill="url(#aui-chart-grad-{label.replace(/\s+/g, '-')})"
          />
        {/if}

        <path
          d={chartGeometry.path}
          fill="none"
          stroke="var(--color-primary)"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        />
      </svg>
    </div>
  {/if}
</div>
