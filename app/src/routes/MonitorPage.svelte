<script lang="ts">
  import Card from "../lib/ui/Card.svelte";
  import Meter from "../lib/ui/Meter.svelte";
  import Sparkline from "../lib/ui/Sparkline.svelte";

  const sampleTps = [15.2, 19.8, 22.4, 24.1, 23.9, 25.4, 25.0, 26.2, 26.1];
</script>

<div class="p-8 max-w-5xl mx-auto space-y-6 text-left">
  <header class="space-y-1">
    <h1 class="text-2xl font-bold tracking-tight text-text">Hardware & Engine Monitor</h1>
    <p class="text-sm text-text-muted">
      Real-time telemetry from your local GPU, VRAM allocation, and generation performance.
    </p>
  </header>

  <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
    <Card variant="surface" padding="md" class="space-y-4">
      <h2 class="text-base font-semibold text-text">GPU Telemetry</h2>
      <Meter label="VRAM Allocation" used={1850} capacity={8192} unit="MB" />
      <div class="grid grid-cols-2 gap-3 pt-2 text-xs font-mono">
        <div class="p-2.5 rounded bg-surface-sunken">
          <span class="text-text-subtle">Compute Util</span>
          <p class="text-sm font-semibold text-text mt-0.5">14%</p>
        </div>
        <div class="p-2.5 rounded bg-surface-sunken">
          <span class="text-text-subtle">Temperature</span>
          <p class="text-sm font-semibold text-text mt-0.5">52°C</p>
        </div>
      </div>
    </Card>

    <Card variant="surface" padding="md" class="space-y-4">
      <h2 class="text-base font-semibold text-text">Inference Speed</h2>
      <div class="flex items-center justify-between pt-2">
        <span class="text-xs text-text-muted">Tokens / second:</span>
        <Sparkline data={sampleTps} label="Live TPS" />
      </div>
      <div class="grid grid-cols-2 gap-3 pt-2 text-xs font-mono">
        <div class="p-2.5 rounded bg-surface-sunken">
          <span class="text-text-subtle">Avg TTFT</span>
          <p class="text-sm font-semibold text-text mt-0.5">142 ms</p>
        </div>
        <div class="p-2.5 rounded bg-surface-sunken">
          <span class="text-text-subtle">Active Batch</span>
          <p class="text-sm font-semibold text-text mt-0.5">1</p>
        </div>
      </div>
    </Card>
  </div>
</div>
