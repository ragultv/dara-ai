import React, { useEffect, useState } from "react";
import { useSettingsStore } from "../lib/stores/useSettingsStore.ts";
import { SettingsNav, type NavItem } from "../lib/features/settings/SettingsNav.tsx";
import { AppearanceSection } from "../lib/features/settings/AppearanceSection.tsx";
import { EngineSection } from "../lib/features/settings/EngineSection.tsx";
import { DefaultsSection } from "../lib/features/settings/DefaultsSection.tsx";
import { DataSection } from "../lib/features/settings/DataSection.tsx";
import { AboutSection } from "../lib/features/settings/AboutSection.tsx";
import { useServerFacts } from "../lib/features/settings/useServerFacts.ts";

const NAV_ITEMS: NavItem[] = [
  { id: "settings-appearance", label: "Appearance" },
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
  const [tab, setTab] = useState<string>(NAV_ITEMS[0]!.id);
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
    <div className="flex h-full bg-bg text-text">
      {/* Left menu: stays in place while the content on the right scrolls */}
      <aside className="flex w-56 shrink-0 flex-col gap-3 border-r border-border p-4">
        <SettingsNav items={NAV_ITEMS} active={tab} onSelect={setTab} />
        <p
          role="status"
          aria-live="polite"
          className={`px-3 text-[11px] font-medium ${saveState === "error" ? "text-danger" : "text-text-muted"}`}
        >
          {SAVE_LABEL[saveState]}
        </p>
      </aside>

      {/* Right: the selected tab, scrolls on its own */}
      <div className="min-w-0 flex-1 overflow-y-auto px-8 py-6 custom-scrollbar">
        <div className="mx-auto max-w-3xl pb-16 text-left">
          {loadError && (
            <p
              role="alert"
              className="mb-4 rounded-xl border border-danger/30 bg-danger-subtle px-4 py-2.5 text-xs text-danger"
            >
              Your saved settings could not be loaded ({loadError}). Defaults are shown; changes
              cannot be saved until the engine is reachable.
            </p>
          )}
          <div role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`}>
            {tab === "settings-appearance" && <AppearanceSection />}
            {tab === "settings-engine" && <EngineSection facts={facts} />}
            {tab === "settings-defaults" && <DefaultsSection />}
            {tab === "settings-data" && <DataSection facts={facts} />}
            {tab === "settings-about" && <AboutSection facts={facts} />}
          </div>
        </div>
      </div>
    </div>
  );
};
