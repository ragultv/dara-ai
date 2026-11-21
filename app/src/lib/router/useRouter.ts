import { create } from "zustand";

export type AppRoute =
  "chat" | "models" | "monitor" | "settings" | "kit" | "options" | "artifacts" | "artifact" | "connectors";

interface RouterState {
  current: AppRoute;
  optionsModelId: string;
  /** The open artifact (route "artifact", hash #/artifacts/<id>). */
  artifactId: string | null;
  /** navigate("options", modelId) opens that model's options; navigate("artifact", id) opens that artifact. */
  navigate: (route: AppRoute, arg?: string) => void;
}

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

function parseHash(): { route: AppRoute; modelId: string; artifactId: string | null } {
  if (typeof window === "undefined") {
    return { route: "chat", modelId: "dara-nano", artifactId: null };
  }
  const raw = window.location.hash.replace(/^#\/?/, "");
  const parts = raw.split("?");
  const rawPath = parts[0] || "";
  const path = rawPath.toLowerCase();
  let modelId = "dara-nano";

  if (parts[1]) {
    const params = new URLSearchParams(parts[1]);
    const m = params.get("model");
    if (m) modelId = m;
  }

  // #/artifacts/<uuid> opens one artifact
  const artifactMatch = /^artifacts\/([^/]+)$/i.exec(rawPath);
  if (artifactMatch && UUID_RE.test(artifactMatch[1]!)) {
    return { route: "artifact", modelId, artifactId: artifactMatch[1]!.toLowerCase() };
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
    case "artifacts":
      route = "artifacts";
      break;
    case "connectors":
      route = "connectors";
      break;
    case "chat":
    default:
      route = "chat";
      break;
  }

  return { route, modelId, artifactId: null };
}

export const useRouter = create<RouterState>((set) => {
  const initial = parseHash();

  if (typeof window !== "undefined") {
    window.addEventListener("hashchange", () => {
      const parsed = parseHash();
      set({
        current: parsed.route,
        optionsModelId: parsed.modelId,
        artifactId: parsed.artifactId
      });
    });
  }

  return {
    current: initial.route,
    optionsModelId: initial.modelId,
    artifactId: initial.artifactId,
    navigate: (route: AppRoute, arg?: string) => {
      const targetModel = (route === "options" && arg) || "dara-nano";
      const artifactId = route === "artifact" && arg ? arg : null;
      set({
        current: route,
        artifactId,
        ...(route === "options" && arg ? { optionsModelId: arg } : {})
      });

      if (typeof window !== "undefined") {
        if (route === "chat") {
          window.location.hash = "#/";
        } else if (route === "options") {
          window.location.hash = `#/options?model=${targetModel}`;
        } else if (route === "artifact" && artifactId) {
          window.location.hash = `#/artifacts/${artifactId}`;
        } else {
          window.location.hash = `#/${route === "artifact" ? "artifacts" : route}`;
        }
      }
    }
  };
});
