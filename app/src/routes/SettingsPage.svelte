<script lang="ts">
  import Card from "../lib/ui/Card.svelte";
  import Toggle from "../lib/ui/Toggle.svelte";
  import SegmentedControl from "../lib/ui/SegmentedControl.svelte";
  import Input from "../lib/ui/Input.svelte";
  import { themeStore, type ThemeMode } from "../lib/stores/theme.svelte.ts";
  import { serverStore } from "../lib/stores/server.svelte.ts";

  let startAtLogin = $state(false);
  let minimizeToTray = $state(true);
  let networkAccess = $state(false);

  const themeOptions = [
    { value: "system", label: "System" },
    { value: "light", label: "Light" },
    { value: "dark", label: "Dark" }
  ];

  function handleThemeChange(val: string) {
    themeStore.setMode(val as ThemeMode);
  }
</script>

<div class="p-8 max-w-4xl mx-auto space-y-6 text-left">
  <header class="space-y-1">
    <h1 class="text-2xl font-bold tracking-tight text-text">Settings</h1>
    <p class="text-sm text-text-muted">
      Configure application appearance, server, and preferences.
    </p>
  </header>

  <div class="space-y-6">
    <Card variant="surface" padding="md" class="space-y-4">
      <h2 class="text-base font-semibold text-text">Appearance</h2>
      <SegmentedControl
        label="Application Theme"
        options={themeOptions}
        value={themeStore.mode}
        onchange={handleThemeChange}
      />
    </Card>

    <Card variant="surface" padding="md" class="space-y-4">
      <h2 class="text-base font-semibold text-text">System Integration</h2>
      <Toggle
        label="Start Dara at Windows login"
        description="Launch minimized to system tray on startup"
        bind:checked={startAtLogin}
      />
      <Toggle
        label="Minimize to System Tray"
        description="Keep background sidecar engine running when closing the window"
        bind:checked={minimizeToTray}
      />
    </Card>

    <Card variant="surface" padding="md" class="space-y-4">
      <h2 class="text-base font-semibold text-text">Server & Network Security</h2>
      <Input
        label="Engine Endpoint"
        value={serverStore.baseUrl}
        readonly={true}
        hint="Managed local loopback sidecar connection"
      />
      <Toggle
        label="Allow other devices on local network"
        description="Off by default for defense-in-depth security. When enabled, requires strict API authentication."
        bind:checked={networkAccess}
      />
    </Card>
  </div>
</div>
