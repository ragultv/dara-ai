import React from "react";
import { useThemeStore, type ThemeChoice } from "../../stores/useThemeStore.ts";
import { SegmentedControl } from "../../ui/SegmentedControl.tsx";
import { SettingsSection } from "./SettingsSection.tsx";

const THEME_OPTIONS = [
  { value: "system", label: "System" },
  { value: "light", label: "Light" },
  { value: "dark", label: "Dark" }
];

export const GeneralSection: React.FC = () => {
  const theme = useThemeStore((s) => s.theme);
  const setTheme = useThemeStore((s) => s.setTheme);
  return (
    <SettingsSection id="settings-general" icon="sliders" title="General">
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
    </SettingsSection>
  );
};
