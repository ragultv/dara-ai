<script lang="ts">
  import Card from "../lib/ui/Card.svelte";
  import Badge from "../lib/ui/Badge.svelte";
  import Button from "../lib/ui/Button.svelte";
</script>

<div class="p-8 max-w-5xl mx-auto space-y-6 text-left">
  <header class="space-y-1">
    <h1 class="text-2xl font-bold tracking-tight text-text">Models</h1>
    <p class="text-sm text-text-muted">
      Select and manage local models optimized for your GPU hardware.
    </p>
  </header>

  <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
    <Card variant="surface" padding="md" class="space-y-4">
      <div class="flex items-center justify-between">
        <h2 class="text-base font-semibold text-text">Dara Nano</h2>
        <Badge variant="success">Recommended</Badge>
      </div>
      <p class="text-xs text-text-muted">
        1.2B parameters. Ultra-fast, fits easily on laptops with 4 GB VRAM.
      </p>
      <div class="text-xs text-text-subtle space-y-1 font-mono">
        <div>Download: ~0.8 GB</div>
        <div>Context: 32K tokens</div>
      </div>
      <Button variant="primary" size="sm" class="w-full">Download (0.8 GB)</Button>
    </Card>

    <Card variant="surface" padding="md" class="space-y-4">
      <div class="flex items-center justify-between">
        <h2 class="text-base font-semibold text-text">Dara Core 4B</h2>
        <Badge variant="purple">Medium</Badge>
      </div>
      <p class="text-xs text-text-muted">
        4B parameters. Balanced reasoning and instruction-following.
      </p>
      <div class="text-xs text-text-subtle space-y-1 font-mono">
        <div>Download: ~2.6 GB</div>
        <div>Context: 32K tokens</div>
      </div>
      <Button variant="secondary" size="sm" class="w-full">Download (2.6 GB)</Button>
    </Card>

    <Card variant="surface" padding="md" class="space-y-4">
      <div class="flex items-center justify-between">
        <h2 class="text-base font-semibold text-text">Dara Pro 9B</h2>
        <Badge variant="neutral">Advanced</Badge>
      </div>
      <p class="text-xs text-text-muted">
        9B parameters. Complex multi-step reasoning, best for 8 GB+ GPUs.
      </p>
      <div class="text-xs text-text-subtle space-y-1 font-mono">
        <div>Download: ~5.8 GB</div>
        <div>Context: 64K tokens</div>
      </div>
      <Button variant="secondary" size="sm" class="w-full">Download (5.8 GB)</Button>
    </Card>
  </div>
</div>
