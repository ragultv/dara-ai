import { create } from "zustand";

export type ThemeChoice = "system" | "light" | "dark";
export type ThemeMode = ThemeChoice;

interface ThemeState {
  theme: ThemeChoice;
  mode: ThemeChoice;
  effectiveTheme: "light" | "dark";
  /** The user picked a theme: apply it, cache it and queue the save. */
  setTheme: (theme: ThemeChoice) => void;
  setMode: (mode: ThemeMode) => void;
  /** A theme read from the database: apply and cache it without writing it back. */
  hydrate: (theme: ThemeChoice) => void;
  init: () => void;
}

// localStorage is only an instant-load cache so the page does not flash the wrong theme;
// the saved value lives in the server database under the flat key "theme".
const STORAGE_KEY = "dara_theme_choice";

let persister: ((choice: ThemeChoice) => void) | null = null;

/** The settings store registers itself here so every theme write goes through its debounced queue. */
export function registerThemePersister(fn: (choice: ThemeChoice) => void): void {
  persister = fn;
}

function getSystemTheme(): "light" | "dark" {
  if (typeof window === "undefined" || !window.matchMedia) return "dark";
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

function applyThemeToDom(effective: "light" | "dark") {
  if (typeof document === "undefined") return;
  const root = document.documentElement;
  root.setAttribute("data-theme", effective);
  if (effective === "dark") {
    root.classList.add("dark");
  } else {
    root.classList.remove("dark");
  }
}

function cacheChoice(choice: ThemeChoice): void {
  try {
    localStorage.setItem(STORAGE_KEY, choice);
  } catch {
    // ignore: the cache is optional
  }
}

let mediaQuery: MediaQueryList | null = null;
let mediaListener: (() => void) | null = null;

export const useThemeStore = create<ThemeState>((set, get) => {
  function apply(choice: ThemeChoice): void {
    const effective = choice === "system" ? getSystemTheme() : choice;
    set({ theme: choice, mode: choice, effectiveTheme: effective });
    applyThemeToDom(effective);
  }

  return {
    theme: "system",
    mode: "system",
    effectiveTheme: "dark",

    init: () => {
      let saved: ThemeChoice = "system";
      try {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored === "system" || stored === "light" || stored === "dark") saved = stored;
      } catch {
        // ignore
      }
      apply(saved);

      if (typeof window !== "undefined" && typeof window.matchMedia === "function") {
        // init can run twice (StrictMode): never stack listeners.
        if (mediaQuery && mediaListener) mediaQuery.removeEventListener?.("change", mediaListener);
        mediaQuery = window.matchMedia("(prefers-color-scheme: light)");
        mediaListener = () => {
          if (get().theme === "system") apply("system");
        };
        mediaQuery.addEventListener?.("change", mediaListener);
      }
    },

    setTheme: (choice) => {
      cacheChoice(choice);
      apply(choice);
      persister?.(choice);
    },

    setMode: (mode) => {
      get().setTheme(mode);
    },

    hydrate: (choice) => {
      cacheChoice(choice);
      apply(choice);
    }
  };
});
