import React, { useEffect } from "react";
import { useSettingsStore } from "../lib/stores/useSettingsStore.ts";
import { SettingsNav, type NavItem } from "../lib/features/settings/SettingsNav.tsx";
import { GeneralSection } from "../lib/features/settings/GeneralSection.tsx";
import { EngineSection } from "../lib/features/settings/EngineSection.tsx";
import { DefaultsSection } from "../lib/features/settings/DefaultsSection.tsx";
import { DataSection } from "../lib/features/settings/DataSection.tsx";
import { AboutSection } from "../lib/features/settings/AboutSection.tsx";
import { useServerFacts } from "../lib/features/settings/useServerFacts.ts";

const NAV_ITEMS: NavItem[] = [
  { id: "settings-general", label: "General" },
  { id: "settings-engine", label: "Engine" },
  { id: "settings-defaults", label: "Defaults" },
  { id: "settings-data", label: "Data" },
  { id: "settings-about", label: "About" }
];

const SAVE_LABEL = {
  idle: "",
  saving: "Saving...",
  saved: "Saved",
  error: "Not saved"
} as const;

export const SettingsPage: React.FC = () => {
  const facts = useServerFacts();
  const saveState = useSettingsStore((s) => s.saveState);
  const loadError = useSettingsStore((s) => s.error);
  const flush = useSettingsStore((s) => s.flush);

  // A pending edit is written when the user leaves the page.
  useEffect(() => {
    return () => {
      void flush();
    };
  }, [flush]);

  return (
    <div className="h-full overflow-y-auto bg-bg px-6 py-6 text-text custom-scrollbar">
      <div className="mx-auto max-w-5xl pb-16 text-left">
        <header className="mb-6 flex items-center justify-between gap-3 border-b border-border pb-4">
          <div className="space-y-0.5">
            <h1 className="text-xl font-bold tracking-tight">Settings</h1>
            <p className="text-xs text-text-muted">Changes are saved automatically.</p>
          </div>
          <p
            role="status"
            aria-live="polite"
            className={`text-xs font-medium ${saveState === "error" ? "text-danger" : "text-text-muted"}`}
          >
            {SAVE_LABEL[saveState]}
          </p>
        </header>

        {loadError && (
          <p
            role="alert"
            className="mb-4 rounded-xl border border-danger/30 bg-danger-subtle px-4 py-2.5 text-xs text-danger"
          >
            Your saved settings could not be loaded ({loadError}). Defaults are shown; changes
            cannot be saved until the engine is reachable.
          </p>
        )}

        <div className="flex gap-8">
          <SettingsNav items={NAV_ITEMS} />
          <div className="min-w-0 flex-1 space-y-10">
            <GeneralSection />
            <EngineSection facts={facts} />
            <DefaultsSection />
            <DataSection facts={facts} />
            <AboutSection facts={facts} />
          </div>
        </div>
      </div>
    </div>
  );
};
