import { create } from "zustand";

export type AppRoute = "chat" | "models" | "monitor" | "settings" | "kit" | "options";

interface RouterState {
  current: AppRoute;
  optionsModelId: string;
  navigate: (route: AppRoute, modelId?: string) => void;
}

function parseHash(): { route: AppRoute; modelId: string } {
  if (typeof window === "undefined") {
    return { route: "chat", modelId: "dara-nano" };
  }
  const raw = window.location.hash.replace(/^#\/?/, "");
  const parts = raw.split("?");
  const path = (parts[0] || "").toLowerCase();
  let modelId = "dara-nano";

  if (parts[1]) {
    const params = new URLSearchParams(parts[1]);
    const m = params.get("model");
    if (m) modelId = m;
  }

  let route: AppRoute = "chat";
  switch (path) {
    case "models":
      route = "models";
      break;
    case "options":
      route = "options";
      break;
    case "monitor":
      route = "monitor";
      break;
    case "settings":
      route = "settings";
      break;
    case "kit":
      route = "kit";
      break;
    case "chat":
    default:
      route = "chat";
      break;
  }

  return { route, modelId };
}

export const useRouter = create<RouterState>((set) => {
  const initial = parseHash();

  if (typeof window !== "undefined") {
    window.addEventListener("hashchange", () => {
      const parsed = parseHash();
      set({ current: parsed.route, optionsModelId: parsed.modelId });
    });
  }

  return {
    current: initial.route,
    optionsModelId: initial.modelId,
    navigate: (route: AppRoute, modelId?: string) => {
      const targetModel = modelId || "dara-nano";
      set({ current: route, ...(modelId ? { optionsModelId: modelId } : {}) });

      if (typeof window !== "undefined") {
        if (route === "chat") {
          window.location.hash = "#/";
        } else if (route === "options") {
          window.location.hash = `#/options?model=${targetModel}`;
        } else {
          window.location.hash = `#/${route}`;
        }
      }
    }
  };
});
