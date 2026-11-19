export type AppRoute = "chat" | "models" | "monitor" | "settings" | "kit" | "options";

class Router {
  current = $state<AppRoute>("chat");
  optionsModelId = $state<string>("dara-nano");

  constructor() {
    if (typeof window !== "undefined") {
      this.syncFromHash();
      window.addEventListener("hashchange", () => {
        this.syncFromHash();
      });
    }
  }

  private syncFromHash(): void {
    const raw = window.location.hash.replace(/^#\/?/, "");
    const parts = raw.split("?");
    const path = parts[0]?.toLowerCase() ?? "";

    if (parts[1]) {
      const params = new URLSearchParams(parts[1]);
      const m = params.get("model");
      if (m) this.optionsModelId = m;
    }

    switch (path) {
      case "models":
        this.current = "models";
        break;
      case "options":
        this.current = "options";
        break;
      case "monitor":
        this.current = "monitor";
        break;
      case "settings":
        this.current = "settings";
        break;
      case "kit":
        this.current = "kit";
        break;
      case "chat":
      case "":
      default:
        this.current = "chat";
        break;
    }
  }

  navigate(route: AppRoute, modelId?: string): void {
    this.current = route;
    if (modelId) {
      this.optionsModelId = modelId;
    }
    if (typeof window !== "undefined") {
      if (route === "chat") {
        window.location.hash = "#/";
      } else if (route === "options") {
        window.location.hash = `#/options?model=${this.optionsModelId}`;
      } else {
        window.location.hash = `#/${route}`;
      }
    }
  }
}

export const router = new Router();
