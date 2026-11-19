<script lang="ts">
  import { onMount } from "svelte";
  import Toggle from "../lib/ui/Toggle.svelte";
  import SegmentedControl from "../lib/ui/SegmentedControl.svelte";
  import Input from "../lib/ui/Input.svelte";
  import Textarea from "../lib/ui/Textarea.svelte";
  import Slider from "../lib/ui/Slider.svelte";
  import Button from "../lib/ui/Button.svelte";
  import Dialog from "../lib/ui/Dialog.svelte";
  import Badge from "../lib/ui/Badge.svelte";
  import Icon from "../lib/ui/Icon.svelte";
  import { themeStore, type ThemeMode } from "../lib/stores/theme.svelte.ts";
  import { settingsStore } from "../lib/stores/settings.svelte.ts";
  import { chatStore } from "../lib/stores/chat.svelte.ts";
  import { apiClient } from "../lib/api/client.ts";
  import { toast } from "../lib/ui/Toast.svelte";
  import type { VersionInfo } from "../lib/api/types.ts";

  let versionInfo = $state<VersionInfo | null>(null);
  let isApiKeyRevealed = $state<boolean>(false);
  let isDeleteModalOpen = $state<boolean>(false);

  const themeOptions = [
    { value: "system", label: "System" },
    { value: "light", label: "Light" },
    { value: "dark", label: "Dark" }
  ];

  const kvOptions = [
    { value: "auto", label: "Auto" },
    { value: "q8", label: "INT8" },
    { value: "k8v4", label: "K8V4" },
    { value: "q4", label: "Q4" }
  ];

  const flashOptions = [
    { value: "auto", label: "Auto" },
    { value: "on", label: "Enabled" },
    { value: "off", label: "Disabled" }
  ];

  onMount(() => {
    void loadVersion();
  });

  async function loadVersion() {
    try {
      versionInfo = await apiClient.getVersion();
    } catch {
      // optional
    }
  }

  function handleThemeChange(val: string) {
    themeStore.setMode(val as ThemeMode);
  }

  async function copyApiKey() {
    try {
      await navigator.clipboard.writeText(settingsStore.server.apiKey);
      toast.show("API key copied to clipboard", "success");
    } catch {
      toast.show("Failed to copy API key", "danger");
    }
  }

  function handleRegenerateKey() {
    settingsStore.regenerateApiKey();
    toast.show("Generated new API key", "info");
  }

  function exportConversations() {
    try {
      const dataStr =
        "data:text/json;charset=utf-8," +
        encodeURIComponent(JSON.stringify(chatStore.conversations, null, 2));
      const downloadAnchor = document.createElement("a");
      downloadAnchor.setAttribute("href", dataStr);
      downloadAnchor.setAttribute("download", `dara-conversations-${Date.now()}.json`);
      document.body.appendChild(downloadAnchor);
      downloadAnchor.click();
      downloadAnchor.remove();
      toast.show("Conversations exported", "success");
    } catch {
      toast.show("Export failed", "danger");
    }
  }

  function handleDeleteAllConversations() {
    chatStore.clearAllConversations();
    isDeleteModalOpen = false;
    toast.show("All conversations deleted", "info");
  }

  function handleResetDefaults() {
    settingsStore.reset();
    toast.show("Settings restored to factory defaults", "info");
  }
</script>

<div class="h-full overflow-y-auto px-6 py-8 custom-scrollbar bg-bg text-text">
  <div class="max-w-3xl mx-auto space-y-8 text-left pb-16">
    <!-- Header -->
    <header class="flex items-center justify-between pb-6 border-b border-border">
      <div class="space-y-1">
        <h1 class="text-xl font-bold tracking-tight text-text">Settings</h1>
        <p class="text-xs text-text-muted">
          Manage application theme, local engine configuration, and default inference behavior.
        </p>
      </div>

      <button
        type="button"
        onclick={handleResetDefaults}
        class="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border bg-surface hover:bg-surface-raised text-xs font-semibold text-text-muted hover:text-text transition-colors cursor-pointer shadow-xs"
      >
        <Icon name="refresh-cw" size={12} />
        <span>Reset Defaults</span>
      </button>
    </header>

    <!-- 1. Appearance & Theme -->
    <section class="space-y-3" aria-labelledby="heading-appearance">
      <div class="flex items-center gap-2">
        <Icon name="eye" size={15} class="text-primary" />
        <h2 id="heading-appearance" class="text-xs font-bold text-text uppercase tracking-wider">
          Appearance
        </h2>
      </div>

      <div class="p-5 rounded-2xl bg-surface border border-border space-y-4 shadow-xs">
        <SegmentedControl
          label="Application Theme"
          options={themeOptions}
          value={themeStore.mode}
          onchange={handleThemeChange}
        />
      </div>
    </section>

    <!-- 2. System Integration -->
    <section class="space-y-3" aria-labelledby="heading-system">
      <div class="flex items-center gap-2">
        <Icon name="activity" size={15} class="text-primary" />
        <h2 id="heading-system" class="text-xs font-bold text-text uppercase tracking-wider">
          System Integration
        </h2>
      </div>

      <div class="p-5 rounded-2xl bg-surface border border-border space-y-4 shadow-xs">
        <Toggle
          label="Start Dara at Windows login"
          description="Launch minimized to system tray when your PC starts."
          checked={settingsStore.general.startAtLogin}
          onchange={(val) => settingsStore.updateGeneral({ startAtLogin: val })}
        />
        <div class="border-t border-border/40 pt-3">
          <Toggle
            label="Minimize to system tray"
            description="Closing the window keeps the engine active in the background for instant replies."
            checked={settingsStore.general.minimizeToTray}
            onchange={(val) => settingsStore.updateGeneral({ minimizeToTray: val })}
          />
        </div>
      </div>
    </section>

    <!-- 3. Local Server & Network -->
    <section class="space-y-3" aria-labelledby="heading-server">
      <div class="flex items-center gap-2">
        <Icon name="database" size={15} class="text-primary" />
        <h2 id="heading-server" class="text-xs font-bold text-text uppercase tracking-wider">
          Local Engine & Server
        </h2>
      </div>

      <div class="p-5 rounded-2xl bg-surface border border-border space-y-5 shadow-xs">
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <Input
            label="Server Port"
            type="number"
            value={String(settingsStore.server.port)}
            oninput={(e) =>
              settingsStore.updateServer({
                port: parseInt((e.target as HTMLInputElement).value, 10) || 11435
              })}
            hint="Default: 11435 (HTTP)"
          />

          <Input
            label="Models Directory"
            type="text"
            value={settingsStore.server.modelsDir}
            oninput={(e) =>
              settingsStore.updateServer({ modelsDir: (e.target as HTMLInputElement).value })}
            hint="Path where weights and projections are stored"
          />
        </div>

        <!-- Network Exposure Switch -->
        <div class="pt-4 border-t border-border/50 space-y-4">
          <Toggle
            label="Allow other devices on local network"
            description="Permit phones, tablets, and other devices on your Wi-Fi/LAN to connect."
            checked={settingsStore.server.allowNetworkAccess}
            onchange={(val) => settingsStore.updateServer({ allowNetworkAccess: val })}
          />

          {#if settingsStore.server.allowNetworkAccess}
            <div
              role="alert"
              class="p-3.5 rounded-xl bg-warning-subtle border border-warning text-xs text-warning space-y-1.5"
            >
              <p class="font-bold flex items-center space-x-1.5">
                <span>⚠️ Local Network Security</span>
              </p>
              <p>
                Exposing this port allows any device on your local network to send inference requests.
                Ensure remote clients authenticate with the Bearer API key below.
              </p>
            </div>

            <!-- API Key Box -->
            <div class="p-4 rounded-xl bg-surface-sunken border border-border space-y-3">
              <div class="flex items-center justify-between">
                <span class="text-xs font-semibold text-text">Engine API Key</span>
                <div class="flex items-center gap-2">
                  <button
                    type="button"
                    onclick={() => (isApiKeyRevealed = !isApiKeyRevealed)}
                    class="text-xs text-text-muted hover:text-text cursor-pointer"
                  >
                    {isApiKeyRevealed ? "Hide" : "Reveal"}
                  </button>
                  <Button variant="secondary" size="sm" onclick={copyApiKey}>Copy</Button>
                  <Button variant="ghost" size="sm" onclick={handleRegenerateKey}>Regenerate</Button>
                </div>
              </div>

              <div class="font-mono text-xs p-2 rounded-lg bg-surface border border-border text-text select-all truncate">
                {isApiKeyRevealed ? settingsStore.server.apiKey : "••••••••••••••••••••••••••••••••"}
              </div>
            </div>
          {/if}
        </div>
      </div>
    </section>

    <!-- 4. Generation Defaults -->
    <section class="space-y-3" aria-labelledby="heading-inference">
      <div class="flex items-center gap-2">
        <Icon name="sliders-horizontal" size={15} class="text-primary" />
        <h2 id="heading-inference" class="text-xs font-bold text-text uppercase tracking-wider">
          Inference & Generation Defaults
        </h2>
      </div>

      <div class="p-5 rounded-2xl bg-surface border border-border space-y-5 shadow-xs">
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <SegmentedControl
            label="Default K/V Precision"
            options={kvOptions}
            value={settingsStore.inference.defaultKv}
            onchange={(val) =>
              settingsStore.updateInference({ defaultKv: val as "auto" | "q8" | "k8v4" | "q4" })}
          />

          <SegmentedControl
            label="Default Flash Drafter"
            options={flashOptions}
            value={settingsStore.inference.defaultFlash}
            onchange={(val) =>
              settingsStore.updateInference({ defaultFlash: val as "auto" | "on" | "off" })}
          />
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2 border-t border-border/40">
          <Input
            label="VRAM Reserve (MB)"
            type="number"
            value={String(settingsStore.inference.vramReserveMb)}
            oninput={(e) =>
              settingsStore.updateInference({
                vramReserveMb: parseInt((e.target as HTMLInputElement).value, 10) || 64
              })}
            hint="Reserved for Windows display compositor"
          />

          <Input
            label="Default Max Tokens"
            type="number"
            value={String(settingsStore.inference.defaultMaxTokens)}
            oninput={(e) =>
              settingsStore.updateInference({
                defaultMaxTokens: parseInt((e.target as HTMLInputElement).value, 10) || 4096
              })}
            hint="Maximum completion tokens per turn"
          />
        </div>

        <div class="space-y-2 pt-2 border-t border-border/40">
          <Slider
            label="Sampling Temperature"
            min={0.0}
            max={2.0}
            step={0.1}
            value={settingsStore.inference.defaultTemperature}
            oninput={(val: number) => settingsStore.updateInference({ defaultTemperature: val })}
          />
          <p class="text-[11px] text-text-subtle">
            Lower values (0.2) yield focused deterministic answers; higher values (0.8) increase creativity.
          </p>
        </div>

        <div class="pt-2 border-t border-border/40">
          <Toggle
            label="Enable Multi-modal Vision Support"
            description="Allow image attachments when supported by the loaded model architecture."
            checked={settingsStore.inference.visionEnabled}
            onchange={(val) => settingsStore.updateInference({ visionEnabled: val })}
          />
        </div>

        <div class="space-y-1.5 pt-2 border-t border-border/40">
          <label for="system-prompt-input" class="text-xs font-semibold text-text">Default System Prompt</label>
          <Textarea
            id="system-prompt-input"
            value={settingsStore.inference.systemPrompt}
            rows={3}
            oninput={(e) =>
              settingsStore.updateInference({
                systemPrompt: (e.target as HTMLTextAreaElement).value
              })}
          />
        </div>
      </div>
    </section>

    <!-- 5. Data & Storage -->
    <section class="space-y-3" aria-labelledby="heading-data">
      <div class="flex items-center gap-2">
        <Icon name="database" size={15} class="text-primary" />
        <h2 id="heading-data" class="text-xs font-bold text-text uppercase tracking-wider">
          Data & Local Storage
        </h2>
      </div>

      <div class="p-5 rounded-2xl bg-surface border border-border shadow-xs">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h3 class="text-sm font-semibold text-text">Conversation History</h3>
            <p class="text-xs text-text-muted mt-0.5">
              {chatStore.conversations.length} saved conversation{chatStore.conversations.length === 1
                ? ""
                : "s"} stored locally on your machine.
            </p>
          </div>

          <div class="flex items-center gap-2">
            <Button variant="secondary" size="sm" onclick={exportConversations}>Export JSON</Button>
            <Button
              variant="danger"
              size="sm"
              disabled={chatStore.conversations.length === 0}
              onclick={() => (isDeleteModalOpen = true)}
            >
              Clear History
            </Button>
          </div>
        </div>
      </div>
    </section>

    <!-- 6. About Dara -->
    <section class="space-y-3" aria-labelledby="heading-about">
      <div class="flex items-center gap-2">
        <Icon name="info" size={15} class="text-primary" />
        <h2 id="heading-about" class="text-xs font-bold text-text uppercase tracking-wider">
          About Dara
        </h2>
      </div>

      <div class="p-5 rounded-2xl bg-surface border border-border space-y-4 shadow-xs">
        <div class="flex items-center justify-between border-b border-border/40 pb-3">
          <div class="flex items-center gap-2.5">
            <img src="/dara.png" alt="Dara logo" class="w-6 h-6 object-contain shrink-0" />
            <span class="font-bold text-sm text-text">Dara Studio</span>
            <Badge variant="purple" size="sm">v0.1.0</Badge>
          </div>
          <div class="text-xs font-mono text-text-muted">
            Engine API v{versionInfo?.api ?? 1}
          </div>
        </div>

        <div class="grid grid-cols-2 gap-3 text-xs font-mono text-text-subtle">
          <div>
            Engine: <span class="text-text font-semibold">{versionInfo?.version ?? "0.1.0-dev"}</span>
          </div>
          <div>
            Built: <span class="text-text font-semibold">{versionInfo?.built ?? "Nov 19 2026"}</span>
          </div>
        </div>

        <p class="text-xs text-text-muted leading-relaxed">
          Dara is a private, offline AI studio built specifically for high-throughput local LLM and vision inference on graphics hardware.
        </p>
      </div>
    </section>

    <!-- Confirmation Dialog for Deleting All Conversations -->
    <Dialog
      isOpen={isDeleteModalOpen}
      title="Delete All Conversation History?"
      description="This permanently removes all local chat threads and messages from this device. This action cannot be undone."
      onclose={() => (isDeleteModalOpen = false)}
    >
      {#snippet actions()}
        <Button variant="ghost" size="sm" onclick={() => (isDeleteModalOpen = false)}>Cancel</Button>
        <Button variant="danger" size="sm" onclick={handleDeleteAllConversations}>Delete All</Button>
      {/snippet}
    </Dialog>
  </div>
</div>