export type AppRoute = "chat" | "models" | "monitor" | "settings" | "kit";

class Router {
  current = $state<AppRoute>("chat");

  constructor() {
    if (typeof window !== "undefined") {
      this.syncFromHash();
      window.addEventListener("hashchange", () => {
        this.syncFromHash();
      });
    }
  }

  private syncFromHash(): void {
    const raw = window.location.hash.replace(/^#\/?/, "").toLowerCase();
    switch (raw) {
      case "models":
        this.current = "models";
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

  navigate(route: AppRoute): void {
    this.current = route;
    if (typeof window !== "undefined") {
      window.location.hash = route === "chat" ? "#/" : `#/${route}`;
    }
  }
}

export const router = new Router();
