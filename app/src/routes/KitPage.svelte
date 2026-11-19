<script lang="ts">
  import Button from "../lib/ui/Button.svelte";
  import IconButton from "../lib/ui/IconButton.svelte";
  import Input from "../lib/ui/Input.svelte";
  import Textarea from "../lib/ui/Textarea.svelte";
  import Checkbox from "../lib/ui/Checkbox.svelte";
  import Toggle from "../lib/ui/Toggle.svelte";
  import Slider from "../lib/ui/Slider.svelte";
  import Select from "../lib/ui/Select.svelte";
  import SegmentedControl from "../lib/ui/SegmentedControl.svelte";
  import Card from "../lib/ui/Card.svelte";
  import Badge from "../lib/ui/Badge.svelte";
  import Tooltip from "../lib/ui/Tooltip.svelte";
  import Popover from "../lib/ui/Popover.svelte";
  import Menu from "../lib/ui/Menu.svelte";
  import Dialog from "../lib/ui/Dialog.svelte";
  import { toast } from "../lib/ui/Toast.svelte";
  import ProgressBar from "../lib/ui/ProgressBar.svelte";
  import Skeleton from "../lib/ui/Skeleton.svelte";
  import Tabs from "../lib/ui/Tabs.svelte";
  import EmptyState from "../lib/ui/EmptyState.svelte";
  import ErrorState from "../lib/ui/ErrorState.svelte";
  import Meter from "../lib/ui/Meter.svelte";
  import Sparkline from "../lib/ui/Sparkline.svelte";
  import CodeBlock from "../lib/ui/CodeBlock.svelte";
  import Markdown from "../lib/ui/Markdown.svelte";

  // Interactive component state
  let inputValue = $state("Sample input");
  let textareaValue = $state("Auto-resizing text block\nSecond line");
  let checkboxChecked = $state(true);
  let toggleChecked = $state(true);
  let sliderValue = $state(42);
  let selectValue = $state("opt1");
  let segmentValue = $state("kv-q4");
  let isDialogOpen = $state(false);
  let activeTab = $state("preview");

  const selectOptions = [
    { value: "opt1", label: "Option 1: Nano (0.8 GB)" },
    { value: "opt2", label: "Option 2: Core 4B (2.6 GB)" },
    { value: "opt3", label: "Option 3: Pro 9B (5.8 GB)", disabled: true }
  ];

  const segmentOptions = [
    { value: "kv-q4", label: "Q4_0", description: "High quality" },
    { value: "kv-q8", label: "Q8_0", description: "Best quality" },
    { value: "kv-fp16", label: "FP16", description: "Reference" }
  ];

  const menuItems = [
    { id: "1", label: "Download model", action: () => toast.show("Downloading model...", "info") },
    { id: "2", label: "Copy model ID", action: () => toast.show("Copied ID", "success") },
    {
      id: "3",
      label: "Unload model",
      danger: true,
      action: () => toast.show("Model unloaded", "warning")
    }
  ];

  const sparklineData = [12.4, 18.2, 22.0, 24.5, 23.8, 25.1, 24.9, 26.2, 25.8, 26.5];
  const markdownSample = `### Markdown Feature Support
Here is **bold text**, *italic*, and a [link](https://example.com).
- First list item
- Second item with \`inline_code()\`

> A quote block emphasizing clean local execution.`;

  const codeSample = `// Dara sidecar IPC example
const connection = await invoke("get_server_connection");
console.log("Connected to", connection.baseUrl);`;
</script>

<div class="p-8 max-w-5xl mx-auto space-y-10 text-left">
  <!-- Kit Header -->
  <header class="space-y-2 border-b border-border pb-6">
    <div class="flex items-center space-x-3">
      <h1 class="text-2xl font-bold tracking-tight text-text">Dara Component Kit</h1>
      <Badge variant="purple" size="md">Design System v1.0</Badge>
    </div>
    <p class="text-sm text-text-muted">
      Complete showcase of accessible, token-backed Svelte 5 components with WCAG 2.2 AA contrast.
    </p>
  </header>

  <!-- 1. Buttons -->
  <section class="space-y-4" aria-labelledby="section-buttons">
    <h2 id="section-buttons" class="text-lg font-semibold text-text">Buttons & IconButtons</h2>
    <div class="flex flex-wrap items-center gap-3">
      <Button variant="primary">Primary Button</Button>
      <Button variant="secondary">Secondary Button</Button>
      <Button variant="ghost">Ghost Button</Button>
      <Button variant="danger">Danger Button</Button>
      <Button variant="primary" size="sm">Small Primary</Button>
      <Button variant="primary" isLoading={true}>Loading</Button>
      <Button variant="secondary" disabled={true}>Disabled</Button>

      <IconButton aria-label="Search models" variant="secondary">
        <svg
          class="w-4 h-4"
          viewBox="0 0 20 20"
          fill="none"
          stroke="currentColor"
          stroke-width="1.5"
          aria-hidden="true"
        >
          <path
            d="M9 17A8 8 0 1 0 9 1a8 8 0 0 0 0 16zM19 19l-4.35-4.35"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
        </svg>
      </IconButton>
    </div>
  </section>

  <!-- 2. Badges -->
  <section class="space-y-4" aria-labelledby="section-badges">
    <h2 id="section-badges" class="text-lg font-semibold text-text">Badges (Quality & Status)</h2>
    <div class="flex flex-wrap items-center gap-2">
      <Badge variant="neutral">Neutral (Lower)</Badge>
      <Badge variant="purple">Purple (Medium)</Badge>
      <Badge variant="success">Success (High Quality)</Badge>
      <Badge variant="warning">Warning (Experimental)</Badge>
      <Badge variant="danger">Danger (Error)</Badge>
    </div>
  </section>

  <!-- 3. Form Controls -->
  <section class="space-y-6" aria-labelledby="section-forms">
    <h2 id="section-forms" class="text-lg font-semibold text-text">Form Controls</h2>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
      <Input label="Prompt Prefix" hint="Optional system prompt header" bind:value={inputValue} />
      <Input label="Temperature" error="Must be between 0.0 and 2.0" value="3.5" />
      <Select label="Model Selection" options={selectOptions} bind:value={selectValue} />
      <Slider
        label="Max Output Tokens"
        min={128}
        max={8192}
        step={128}
        unit=" tokens"
        bind:value={sliderValue}
      />
      <Textarea
        label="System Prompt"
        hint="Default context instructions"
        bind:value={textareaValue}
      />
      <div class="space-y-4 pt-2">
        <Checkbox
          label="Stream response tokens"
          description="Render tokens progressively as they arrive"
          bind:checked={checkboxChecked}
        />
        <Toggle
          label="Flash Speculative Drafter"
          description="Accelerate generation by drafting ahead"
          bind:checked={toggleChecked}
        />
      </div>
    </div>

    <div>
      <SegmentedControl
        label="K/V Cache Format"
        options={segmentOptions}
        bind:value={segmentValue}
      />
    </div>
  </section>

  <!-- 4. Feedback & Gauges -->
  <section class="space-y-6" aria-labelledby="section-feedback">
    <h2 id="section-feedback" class="text-lg font-semibold text-text">
      Progress, Meters & Telemetry
    </h2>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
      <ProgressBar label="Model Download (Q4_K_M)" value={68} max={100} showPercent={true} />
      <ProgressBar label="Loading Engine Layers" />
      <Meter label="Context Window" used={14200} capacity={32768} />
      <Meter label="VRAM Allocated (Warning Threshold)" used={7200} capacity={8192} unit="MB" />
    </div>

    <div class="flex items-center space-x-6">
      <div>
        <span class="text-xs font-medium text-text-muted block mb-1">Live Decode Rate</span>
        <Sparkline data={sparklineData} label="Generation tok/s" />
      </div>
      <div>
        <span class="text-xs font-medium text-text-muted block mb-1">Loading Skeleton</span>
        <div class="space-y-1.5 w-48">
          <Skeleton height="h-3.5" />
          <Skeleton height="h-3" width="w-3/4" />
        </div>
      </div>
    </div>
  </section>

  <!-- 5. Cards & Containers -->
  <section class="space-y-4" aria-labelledby="section-cards">
    <h2 id="section-cards" class="text-lg font-semibold text-text">Cards & Tabs</h2>
    <Card variant="surface" padding="md" class="space-y-4">
      <Tabs
        tabs={[
          { id: "preview", label: "Model Card" },
          { id: "specs", label: "Specs & Quantization", badge: "3" },
          { id: "notes", label: "Safety Notes" }
        ]}
        bind:activeTab
      >
        {#snippet children({ activeTab: tab })}
          {#if tab === "preview"}
            <div class="p-2 space-y-2">
              <h3 class="text-base font-semibold text-text">Dara Nano (1.2B)</h3>
              <p class="text-xs text-text-muted">
                Ultra-compact instruction-tuned model fitting laptops with 4 GB VRAM.
              </p>
            </div>
          {:else if tab === "specs"}
            <div class="p-2 text-xs text-text-muted space-y-1">
              <p>Format: GGUF Q4_K_M</p>
              <p>Context: 32K tokens</p>
              <p>VRAM required: ~1.4 GB</p>
            </div>
          {:else}
            <div class="p-2 text-xs text-text-muted">
              Standard local offline processing with strict sandboxing.
            </div>
          {/if}
        {/snippet}
      </Tabs>
    </Card>
  </section>

  <!-- 6. Overlays & Modals -->
  <section class="space-y-4" aria-labelledby="section-overlays">
    <h2 id="section-overlays" class="text-lg font-semibold text-text">Overlays, Menus & Dialogs</h2>
    <div class="flex flex-wrap items-center gap-4">
      <Button variant="secondary" onclick={() => (isDialogOpen = true)}>Open Modal Dialog</Button>

      <Tooltip text="Click to regenerate server API key">
        <Button variant="ghost">Hover for Tooltip</Button>
      </Tooltip>

      <Menu items={menuItems}>
        {#snippet trigger({ isOpen, toggle })}
          <Button variant="secondary" onclick={toggle} aria-expanded={isOpen}>
            Actions Menu ▾
          </Button>
        {/snippet}
      </Menu>

      <Popover>
        {#snippet trigger({ isOpen, toggle })}
          <Button variant="secondary" onclick={toggle} aria-expanded={isOpen}>
            Generation Options
          </Button>
        {/snippet}
        {#snippet children({ close })}
          <div class="space-y-3">
            <h4 class="text-xs font-semibold text-text">Quick Generation Overrides</h4>
            <Toggle label="Vision Support" checked={true} />
            <Button size="sm" variant="primary" onclick={close}>Done</Button>
          </div>
        {/snippet}
      </Popover>

      <Button
        variant="ghost"
        onclick={() => toast.show("Background download completed", "success")}
      >
        Trigger Toast
      </Button>
    </div>

    <Dialog
      isOpen={isDialogOpen}
      title="Delete Model Confirmation"
      description="This will permanently delete the local GGUF weights from your disk."
      onclose={() => (isDialogOpen = false)}
    >
      <p class="text-xs text-text-muted">
        Are you sure you want to remove <strong>dara-core-4b.gguf</strong>? You will need to
        redownload 2.6 GB to use this model again.
      </p>
      {#snippet actions()}
        <Button variant="ghost" size="sm" onclick={() => (isDialogOpen = false)}>Cancel</Button>
        <Button
          variant="danger"
          size="sm"
          onclick={() => {
            isDialogOpen = false;
            toast.show("Model deleted", "danger");
          }}
        >
          Delete Model
        </Button>
      {/snippet}
    </Dialog>
  </section>

  <!-- 7. States & Code -->
  <section class="space-y-6" aria-labelledby="section-states">
    <h2 id="section-states" class="text-lg font-semibold text-text">Content & Error States</h2>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
      <EmptyState
        title="No Models Downloaded"
        description="Choose a recommended model to begin chatting offline on your local GPU."
      >
        {#snippet actions()}
          <Button size="sm">Browse Models</Button>
        {/snippet}
      </EmptyState>

      <ErrorState
        title="Engine Failed to Bind Port"
        message="Address 127.0.0.1:11435 is already in use by another process."
        details="Error: os error 10048 (WSAEADDRINUSE) at bind_socket(11435)"
        onretry={() => toast.show("Retrying port allocation...", "info")}
      />
    </div>

    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
      <CodeBlock code={codeSample} language="typescript" filename="client.ts" />
      <Card variant="sunken" padding="md">
        <Markdown content={markdownSample} />
      </Card>
    </div>
  </section>
</div>
