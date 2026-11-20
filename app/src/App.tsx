import React, { useState, useEffect, type ReactNode } from "react";
import { useRouter } from "./lib/router/useRouter.ts";
import { useThemeStore } from "./lib/stores/useThemeStore.ts";
import { useServerStore } from "./lib/stores/useServerStore.ts";
import { useModelsStore } from "./lib/stores/useModelsStore.ts";
import { useSettingsStore } from "./lib/stores/useSettingsStore.ts";
import { useChatStore } from "./lib/stores/useChatStore.ts";

import { LeftRail } from "./lib/shell/LeftRail.tsx";
import { Header } from "./lib/shell/Header.tsx";
import { StatusPill } from "./lib/shell/StatusPill.tsx";
import { TitleBar, isTauriWindow } from "./lib/shell/TitleBar.tsx";
import { ToastContainer } from "./lib/ui/Toast.tsx";

import { ChatPage } from "./routes/ChatPage.tsx";
import { ModelsPage } from "./routes/ModelsPage.tsx";
import { MonitorPage } from "./routes/MonitorPage.tsx";
import { OptionsPage } from "./routes/OptionsPage.tsx";
import { SettingsPage } from "./routes/SettingsPage.tsx";
import { Onboarding } from "./routes/Onboarding.tsx";
import { KitPage } from "./routes/KitPage.tsx";

export const App: React.FC = () => {
  const [isRailCollapsed, setIsRailCollapsed] = useState(false);
  const currentRoute = useRouter((s) => s.current);

  const initTheme = useThemeStore((s) => s.init);
  const initServer = useServerStore((s) => s.init);
  const fetchModels = useModelsStore((s) => s.fetchModels);
  const initSettings = useSettingsStore((s) => s.init);
  const loadConversations = useChatStore((s) => s.loadConversations);

  const models = useModelsStore((s) => s.models);
  const hasInstalledModels = models.some((m) => m.dara.status === "installed");
  // first-run setup (download, image support, Flash, load) keeps the onboarding screen up until it finishes
  const setupModelId = useModelsStore((s) => s.setupModelId);

  useEffect(() => {
    initTheme();
    void initServer().catch(() => {
      // Local fallback in case server initialization errors
      void fetchModels();
      void initSettings();
      void loadConversations();
    });
  }, [initTheme, initServer, fetchModels, initSettings, loadConversations]);

  // Pages without their own scroll container scroll inside the viewport instead of overflowing it.
  const scrollable = (page: ReactNode) => (
    <div className="h-full w-full overflow-y-auto custom-scrollbar">{page}</div>
  );

  const renderRoute = () => {
    // If no models are installed and not on kit or settings, show onboarding
    if (
      (models.length > 0 &&
        !hasInstalledModels &&
        currentRoute !== "settings" &&
        currentRoute !== "kit") ||
      setupModelId !== null
    ) {
      return <Onboarding />;
    }

    switch (currentRoute) {
      case "models":
        return scrollable(<ModelsPage />);
      case "monitor":
        return <MonitorPage />;
      case "options":
        return scrollable(<OptionsPage />);
      case "settings":
        return <SettingsPage />;
      case "kit":
        return scrollable(<KitPage />);
      case "chat":
      default:
        return <ChatPage />;
    }
  };

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-bg text-text select-text antialiased">
      {/* VS Code-style title bar (inside the desktop window only) */}
      {isTauriWindow() && <TitleBar />}
      <div className="flex min-h-0 flex-1">
        {/* Left Sidebar Navigation */}
        <LeftRail
          isCollapsed={isRailCollapsed}
          onToggle={() => setIsRailCollapsed(!isRailCollapsed)}
        />

        {/* Main App Container */}
        <div className="flex-1 flex flex-col h-full min-w-0 overflow-hidden relative">
          {/* Top Header */}
          <Header />

          {/* Dynamic Route Viewport */}
          <div className="flex-1 overflow-hidden relative bg-bg">{renderRoute()}</div>

          {/* Bottom Status Pill Footer */}
          <StatusPill />
        </div>
      </div>

      {/* Global Toast Notifications */}
      <ToastContainer />
    </div>
  );
};
