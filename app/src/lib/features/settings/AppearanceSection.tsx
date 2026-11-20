import React from "react";
import { useThemeStore, type ThemeChoice } from "../../stores/useThemeStore.ts";
import {
  CODE_FONTS,
  UI_FONTS,
  useAppearanceStore,
  type FontSize,
  type ReplyFont
} from "../../stores/useAppearanceStore.ts";
import { SegmentedControl } from "../../ui/SegmentedControl.tsx";
import { Select } from "../../ui/Select.tsx";
import { Input } from "../../ui/Input.tsx";
import { Button } from "../../ui/Button.tsx";
import { SettingsSection } from "./SettingsSection.tsx";

const THEME_OPTIONS = [
  { value: "system", label: "System" },
  { value: "light", label: "Light" },
  { value: "dark", label: "Dark" }
];

const SIZE_OPTIONS = [
  { value: "small", label: "Small" },
  { value: "default", label: "Default" },
  { value: "large", label: "Large" }
];

const REPLY_OPTIONS = [
  { value: "serif", label: "Serif" },
  { value: "ui", label: "Same as interface" }
];

const withCustom = (presets: { id: string; label: string }[]) => [
  ...presets.map((p) => ({ value: p.id, label: p.label })),
  { value: "custom", label: "Custom…" }
];

export const AppearanceSection: React.FC = () => {
  const theme = useThemeStore((s) => s.theme);
  const setTheme = useThemeStore((s) => s.setTheme);
  const a = useAppearanceStore();

  return (
    <SettingsSection
      id="settings-appearance"
      icon="palette"
      title="Appearance"
      description="How Dara looks on this computer. Changes apply to the whole app at once."
    >
      <div className="space-y-1.5">
        <SegmentedControl
          label="Theme"
          options={THEME_OPTIONS}
          value={theme}
          onchange={(v) => setTheme(v as ThemeChoice)}
        />
        <p className="text-[11px] text-text-muted">
          System follows your Windows light or dark setting.
        </p>
      </div>

      <div className="space-y-1.5">
        <SegmentedControl
          label="Text size"
          options={SIZE_OPTIONS}
          value={a.fontSize}
          onchange={(v) => a.update({ fontSize: v as FontSize })}
        />
      </div>

      <div className="space-y-1.5">
        <SegmentedControl
          label="Reply font"
          options={REPLY_OPTIONS}
          value={a.replyFont}
          onchange={(v) => a.update({ replyFont: v as ReplyFont })}
        />
        <p className="text-[11px] text-text-muted">
          The font of the assistant's answers. Serif reads like a printed page.
        </p>
      </div>

      <div className="space-y-2">
        <Select
          label="Interface font"
          options={withCustom(UI_FONTS)}
          value={a.uiFont}
          onChange={(e) => a.update({ uiFont: e.target.value })}
          hint="Fonts already installed on this computer."
        />
        {a.uiFont === "custom" && (
          <Input
            label="Font name"
            placeholder="e.g. Segoe UI Variable"
            value={a.uiFontCustom}
            onChange={(e) => a.update({ uiFontCustom: e.target.value })}
            hint="Type the name of any installed font. If it is not found, the default is used."
          />
        )}
      </div>

      <div className="space-y-2">
        <Select
          label="Code font"
          options={withCustom(CODE_FONTS)}
          value={a.codeFont}
          onChange={(e) => a.update({ codeFont: e.target.value })}
          hint="Used for code blocks and numbers."
        />
        {a.codeFont === "custom" && (
          <Input
            label="Font name"
            placeholder="e.g. JetBrains Mono"
            value={a.codeFontCustom}
            onChange={(e) => a.update({ codeFontCustom: e.target.value })}
          />
        )}
      </div>

      <div className="rounded-md border border-border bg-surface-sunken p-4 space-y-2">
        <p className="text-xs font-semibold text-text-muted">Preview</p>
        <p className="text-sm text-text">
          The quick brown fox jumps over the lazy dog. Dara runs your models on your own GPU.
        </p>
        <pre className="font-mono text-xs text-text">
          {"def merge_sort(items):\n    return sorted(items)  # 0123456789"}
        </pre>
      </div>

      <div>
        <Button variant="secondary" size="sm" onClick={() => a.reset()}>
          Reset appearance
        </Button>
      </div>
    </SettingsSection>
  );
};
