export type ThemeMode = "system" | "light" | "dark";
export type ResolvedTheme = "light" | "dark";

const STORAGE_KEY = "dara-theme";

class ThemeStore {
  mode = $state<ThemeMode>("system");
  systemTheme = $state<ResolvedTheme>("light");

  get resolved(): ResolvedTheme {
    return this.mode === "system" ? this.systemTheme : this.mode;
  }

  constructor() {
    if (typeof window !== "undefined") {
      // Load saved preference
      const saved = localStorage.getItem(STORAGE_KEY) as ThemeMode | null;
      if (saved === "light" || saved === "dark" || saved === "system") {
        this.mode = saved;
      }

      // Check system color scheme if matchMedia is supported
      if (typeof window.matchMedia === "function") {
        const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
        this.systemTheme = mediaQuery.matches ? "dark" : "light";

        // Listen for system changes
        mediaQuery.addEventListener("change", (e) => {
          this.systemTheme = e.matches ? "dark" : "light";
          this.applyDom();
        });
      }

      this.applyDom();
    }
  }

  setMode(newMode: ThemeMode): void {
    this.mode = newMode;
    if (typeof window !== "undefined") {
      localStorage.setItem(STORAGE_KEY, newMode);
      this.applyDom();
    }
  }

  private applyDom(): void {
    if (typeof document === "undefined") return;
    const root = document.documentElement;
    if (this.mode === "system") {
      root.removeAttribute("data-theme");
    } else {
      root.setAttribute("data-theme", this.mode);
    }
  }
}

export const themeStore = new ThemeStore();
