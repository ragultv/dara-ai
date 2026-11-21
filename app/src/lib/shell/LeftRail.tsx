import React from "react";
import { useRouter, type AppRoute } from "../router/useRouter.ts";
import { useModelsStore } from "../stores/useModelsStore.ts";
import { Icon } from "../ui/Icon.tsx";

interface LeftRailProps {
  isCollapsed?: boolean;
}

export const LeftRail: React.FC<LeftRailProps> = ({ isCollapsed = false }) => {
  const currentRoute = useRouter((s) => s.current);
  const navigate = useRouter((s) => s.navigate);
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
      route: "artifacts",
      label: "Artifacts",
      icon: "file-text"
    },
    {
      route: "connectors",
      label: "Connectors",
      icon: "plug"
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
        {/* Nav buttons */}
        <nav className="flex flex-col space-y-1" aria-label="Main">
          {navItems.map((item) => {
            const isActive =
              currentRoute === item.route ||
              (item.route === "artifacts" && currentRoute === "artifact");
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
                className={`flex items-center ${isCollapsed ? "justify-center" : "space-x-3 px-3"} py-2.5 rounded-lg text-xs font-semibold transition-all cursor-pointer border ${
                  isActive
                    ? "bg-surface-raised text-text border-transparent font-bold"
                    : "border-transparent text-text-muted hover:text-text hover:bg-surface-raised/70"
                }`}
              >
                <Icon
                  name={item.icon}
                  size={16}
                  className={isActive ? "text-text" : "text-text-muted"}
                />
                {!isCollapsed && <span className="truncate">{item.label}</span>}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Bottom: Settings */}
      <div className="flex flex-col p-3 border-t border-border">
        {(() => {
          const isActive = currentRoute === "settings";
          return (
            <button
              type="button"
              onClick={() => navigate("settings")}
              aria-current={isActive ? "page" : undefined}
              title={isCollapsed ? "Settings" : undefined}
              className={`flex items-center ${isCollapsed ? "justify-center" : "space-x-3 px-3"} py-2.5 rounded-md text-xs font-semibold transition-all cursor-pointer border ${
                isActive
                  ? "bg-surface-raised text-text border-transparent font-bold"
                  : "border-transparent text-text-muted hover:text-text hover:bg-surface-raised/70"
              }`}
            >
              <Icon
                name="settings"
                size={16}
                className={isActive ? "text-text" : "text-text-muted"}
              />
              {!isCollapsed && <span className="truncate">Settings</span>}
            </button>
          );
        })()}
      </div>
    </aside>
  );
};
