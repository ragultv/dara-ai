import React from "react";
import { useRouter, type AppRoute } from "../router/useRouter.ts";
import { useModelsStore } from "../stores/useModelsStore.ts";
import { useThemeStore, type ThemeMode } from "../stores/useThemeStore.ts";
import { Icon } from "../ui/Icon.tsx";

interface LeftRailProps {
  isCollapsed?: boolean;
  onToggle?: () => void;
}

export const LeftRail: React.FC<LeftRailProps> = ({ isCollapsed = false, onToggle }) => {
  const currentRoute = useRouter((s) => s.current);
  const navigate = useRouter((s) => s.navigate);
  const themeMode = useThemeStore((s) => s.mode);
  const setThemeMode = useThemeStore((s) => s.setMode);
  // Options open for the loaded model, or Nano when nothing is loaded.
  const optionsModelId = useModelsStore(
    (s) => s.models.find((m) => m.dara.loaded)?.id ?? "dara-nano"
  );

  const navItems: Array<{ route: AppRoute; label: string; icon: string }> = [
    {
      route: "chat",
      label: "Chat",
      icon: "message-square-plus"
    },
    {
      route: "models",
      label: "Models",
      icon: "box"
    },
    {
      route: "options",
      label: "Options",
      icon: "sliders-horizontal"
    },
    {
      route: "monitor",
      label: "Monitor",
      icon: "activity"
    },
    {
      route: "settings",
      label: "Settings",
      icon: "settings"
    }
  ];

  return (
    <aside
      className={`flex flex-col justify-between border-r border-border bg-bg-subtle transition-all duration-200 select-none shrink-0 ${
        isCollapsed ? "w-16" : "w-56"
      }`}
      aria-label="Sidebar Navigation"
    >
      {/* Top brand and navigation links */}
      <div className="flex flex-col p-3 space-y-4">
        {/* Header with Dara logo & collapse toggle */}
        <div className="flex items-center justify-between px-1.5 py-1">
          <div className="flex items-center space-x-2.5 overflow-hidden">
            <img src="/dara.png" alt="Dara logo" className="w-7 h-7 shrink-0 object-contain" />

            {!isCollapsed && (
              <div className="flex flex-col text-left">
                <span className="text-sm font-bold tracking-tight text-text leading-tight">
                  DARA STUDIO
                </span>
                <span className="text-[10px] text-text-subtle font-medium">
                  Local GPU Inference
                </span>
              </div>
            )}
          </div>

          <button
            type="button"
            aria-label={isCollapsed ? "Expand navigation rail" : "Collapse navigation rail"}
            onClick={onToggle}
            className="p-1 rounded-md text-text-subtle hover:text-text hover:bg-surface-raised cursor-pointer focus-visible:outline-2 focus-visible:outline-focus"
          >
            <Icon name={isCollapsed ? "chevron-right" : "chevron-left"} size={14} />
          </button>
        </div>

        {/* Nav buttons */}
        <nav className="flex flex-col space-y-1" aria-label="Main">
          {navItems.map((item) => {
            const isActive = currentRoute === item.route;
            return (
              <button
                key={item.route}
                type="button"
                onClick={() =>
                  item.route === "options"
                    ? navigate("options", optionsModelId)
                    : navigate(item.route)
                }
                aria-current={isActive ? "page" : undefined}
                title={isCollapsed ? item.label : undefined}
                className={`flex items-center space-x-3 px-3 py-2.5 rounded-lg text-xs font-semibold transition-all cursor-pointer border ${
                  isActive
                    ? "bg-surface-raised text-primary border-primary/30 shadow-xs font-bold"
                    : "border-transparent text-text-muted hover:text-text hover:bg-surface-raised/70"
                }`}
              >
                <Icon
                  name={item.icon}
                  size={16}
                  className={isActive ? "text-primary" : "text-text-muted"}
                />
                {!isCollapsed && <span className="truncate">{item.label}</span>}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Bottom: Theme selector */}
      <div className="flex flex-col p-3 space-y-3 border-t border-border">
        {!isCollapsed && (
          <div className="flex flex-col space-y-1.5 text-left px-1">
            <span className="text-[10px] font-bold text-text-subtle uppercase tracking-wider">
              Theme
            </span>
            <div className="flex p-0.5 rounded-lg bg-surface-sunken border border-border">
              {(["system", "light", "dark"] as ThemeMode[]).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  onClick={() => setThemeMode(mode)}
                  aria-pressed={themeMode === mode}
                  className={`flex-1 py-1 text-center text-xs font-medium rounded-md cursor-pointer transition-colors capitalize ${
                    themeMode === mode
                      ? "bg-surface text-text shadow-xs"
                      : "text-text-muted hover:text-text"
                  }`}
                >
                  {mode}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </aside>
  );
};
