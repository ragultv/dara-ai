import React, { useState } from "react";
import { useRouter } from "../router/useRouter.ts";
import { useModelsStore } from "../stores/useModelsStore.ts";
import { useServerStore } from "../stores/useServerStore.ts";
import { Icon } from "../ui/Icon.tsx";

export const Header: React.FC = () => {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const currentRoute = useRouter((s) => s.current);

  const models = useModelsStore((s) => s.models);
  const loadedModel = useModelsStore((s) => s.getLoadedModel());
  const loadModel = useModelsStore((s) => s.loadModel);
  const serverLoadedModel = useServerStore((s) => s.loadedModel);

  const activeModel =
    loadedModel ?? (serverLoadedModel ? models.find((m) => m.id === serverLoadedModel) : null);

  const routeName = (() => {
    switch (currentRoute) {
      case "chat":
        return "Chat";
      case "models":
        return "Models";
      case "monitor":
        return "Monitor";
      case "options":
        return "Model Options";
      case "settings":
        return "Settings";
      case "kit":
        return "UI Design Kit";
      default:
        return "Dara Studio";
    }
  })();

  const handleSelectModel = async (modelId: string) => {
    setDropdownOpen(false);
    if (activeModel?.id === modelId) return;
    await loadModel(modelId);
  };

  return (
    <header className="h-12 border-b border-border bg-bg-subtle/80 backdrop-blur-md px-5 flex items-center justify-between shrink-0 select-none z-20">
      {/* Left: Page Title / Breadcrumb */}
      <div className="flex items-center gap-2.5">
        <span className="text-xs font-bold text-text-muted uppercase tracking-wider">
          {routeName}
        </span>
      </div>

      {/* Right: Model Switcher Dropdown & Live Status */}
      <div className="flex items-center gap-3">
        <div className="relative">
          <button
            type="button"
            onClick={() => setDropdownOpen(!dropdownOpen)}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-border bg-surface hover:bg-surface-raised text-xs font-semibold text-text transition-all cursor-pointer shadow-xs"
            aria-expanded={dropdownOpen}
            aria-haspopup="true"
          >
            <Icon name="cpu" size={13} className="text-primary" />
            <span className="max-w-[140px] truncate">
              {activeModel?.dara.name ?? (activeModel?.id ? activeModel.id : "Switch Model")}
            </span>
            {activeModel && <span className="w-1.5 h-1.5 rounded-full bg-success" />}
            <Icon
              name="chevron-down"
              size={12}
              className={`text-text-subtle transition-transform ${dropdownOpen ? "rotate-180" : ""}`}
            />
          </button>

          {dropdownOpen && (
            <>
              {/* Backdrop */}
              <div className="fixed inset-0 z-30" onClick={() => setDropdownOpen(false)} />

              {/* Dropdown Menu */}
              <div className="absolute right-0 top-full mt-1.5 w-76 rounded-xl border border-border bg-surface-raised shadow-xl p-1.5 z-40 text-left animate-in fade-in zoom-in-95 duration-100">
                <div className="px-2.5 py-1.5 text-[10px] font-bold text-text-subtle uppercase tracking-wider border-b border-border/50">
                  Available Models
                </div>
                <div className="max-h-60 overflow-y-auto p-1 space-y-0.5">
                  {models.length === 0 ? (
                    <div className="p-3 text-center text-xs text-text-muted italic">
                      No models registered
                    </div>
                  ) : (
                    models.map((m) => {
                      const isLoaded = m.dara.loaded;
                      const isInstalled = m.dara.status === "installed";
                      return (
                        <button
                          key={m.id}
                          type="button"
                          onClick={() => handleSelectModel(m.id)}
                          disabled={!isInstalled}
                          className={`w-full text-left px-2.5 py-2 rounded-lg text-xs transition-colors flex items-center justify-between gap-2 ${
                            isLoaded
                              ? "bg-primary/10 text-primary font-bold"
                              : isInstalled
                                ? "text-text hover:bg-surface cursor-pointer"
                                : "text-text-muted opacity-50 cursor-not-allowed"
                          }`}
                        >
                          <div className="min-w-0 flex-1">
                            <div className="font-semibold truncate text-[12px]">{m.dara.name}</div>
                            <div className="text-[10px] text-text-subtle flex items-center gap-1.5 mt-0.5">
                              <span>{(m.dara.download_size_mb / 1024).toFixed(1)} GB</span>
                              <span>•</span>
                              <span className="font-mono">{m.dara.quant}</span>
                            </div>
                          </div>
                          {isLoaded ? (
                            <span className="px-1.5 py-0.5 text-[9px] rounded-md font-bold bg-primary text-white">
                              Active
                            </span>
                          ) : isInstalled ? (
                            <span className="text-[11px] text-text-subtle">Ready</span>
                          ) : (
                            <span className="text-[11px] text-text-subtle italic">
                              Not installed
                            </span>
                          )}
                        </button>
                      );
                    })
                  )}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </header>
  );
};
