import { create } from "zustand";

// Fonts and text size. These are per-device display preferences, so they are kept in localStorage (the theme is in the settings database).
// Only fonts already on the machine are offered (the app ships no font files); "Custom" lets the user type any installed family.

export interface FontPreset {
  id: string;
  label: string;
  stack: string;
}

const SYSTEM_UI =
  "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif";
const SYSTEM_MONO =
  "'Cascadia Mono', 'SF Mono', Consolas, 'Liberation Mono', Menlo, Courier, monospace";

export const UI_FONTS: FontPreset[] = [
  { id: "system", label: "System default", stack: SYSTEM_UI },
  { id: "segoe", label: "Segoe UI", stack: `'Segoe UI', ${SYSTEM_UI}` },
  { id: "calibri", label: "Calibri", stack: `Calibri, ${SYSTEM_UI}` },
  { id: "arial", label: "Arial", stack: `Arial, ${SYSTEM_UI}` },
  { id: "verdana", label: "Verdana", stack: `Verdana, ${SYSTEM_UI}` },
  { id: "tahoma", label: "Tahoma", stack: `Tahoma, ${SYSTEM_UI}` },
  { id: "georgia", label: "Georgia (serif)", stack: "Georgia, 'Times New Roman', serif" },
  { id: "mono", label: "Monospace", stack: SYSTEM_MONO }
];

export const CODE_FONTS: FontPreset[] = [
  { id: "system", label: "System default", stack: SYSTEM_MONO },
  { id: "cascadia", label: "Cascadia Mono", stack: `'Cascadia Mono', ${SYSTEM_MONO}` },
  { id: "consolas", label: "Consolas", stack: `Consolas, ${SYSTEM_MONO}` },
  { id: "lucida", label: "Lucida Console", stack: `'Lucida Console', ${SYSTEM_MONO}` },
  { id: "courier", label: "Courier New", stack: `'Courier New', ${SYSTEM_MONO}` }
];

export type ReplyFont = "serif" | "ui";
export type FontSize = "small" | "default" | "large";
const SIZE_PERCENT: Record<FontSize, string> = { small: "90%", default: "100%", large: "112.5%" };

export interface AppearanceState {
  uiFont: string; // a preset id, or "custom"
  uiFontCustom: string;
  codeFont: string;
  codeFontCustom: string;
  fontSize: FontSize;
  /** Font of the assistant's replies: a serif (like Claude) or the interface font. */
  replyFont: ReplyFont;
  init: () => void;
  update: (patch: Partial<Omit<AppearanceState, "init" | "update" | "reset">>) => void;
  reset: () => void;
}

const KEY = "dara_appearance_v1";
const DEFAULTS = {
  uiFont: "system",
  uiFontCustom: "",
  codeFont: "system",
  codeFontCustom: ""
} as const;

type Saved = Pick<
  AppearanceState,
  "uiFont" | "uiFontCustom" | "codeFont" | "codeFontCustom" | "fontSize" | "replyFont"
>;

const clean = (name: string): string => name.replace(/[^\w\s\-.,']/g, "").trim();

function stackFor(presets: FontPreset[], id: string, custom: string, fallback: string): string {
  if (id === "custom") {
    const name = clean(custom);
    return name ? `'${name.replace(/'/g, "")}', ${fallback}` : fallback;
  }
  return presets.find((p) => p.id === id)?.stack ?? fallback;
}

function apply(s: Saved): void {
  if (typeof document === "undefined") return;
  const root = document.documentElement;
  root.style.setProperty("--font-ui", stackFor(UI_FONTS, s.uiFont, s.uiFontCustom, SYSTEM_UI));
  root.style.setProperty(
    "--font-mono",
    stackFor(CODE_FONTS, s.codeFont, s.codeFontCustom, SYSTEM_MONO)
  );
  root.style.fontSize = SIZE_PERCENT[s.fontSize];
  if (s.replyFont === "ui") root.style.setProperty("--font-reply", "var(--font-ui)");
  else root.style.removeProperty("--font-reply"); // the serif default in app.css
}

function read(): Saved {
  const base: Saved = { ...DEFAULTS, fontSize: "default", replyFont: "serif" };
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) {
      const p = JSON.parse(raw) as Partial<Saved>;
      return {
        uiFont: typeof p.uiFont === "string" ? p.uiFont : base.uiFont,
        uiFontCustom: typeof p.uiFontCustom === "string" ? p.uiFontCustom : "",
        codeFont: typeof p.codeFont === "string" ? p.codeFont : base.codeFont,
        codeFontCustom: typeof p.codeFontCustom === "string" ? p.codeFontCustom : "",
        replyFont: p.replyFont === "ui" ? "ui" : "serif",
        fontSize:
          p.fontSize === "small" || p.fontSize === "large" || p.fontSize === "default"
            ? p.fontSize
            : "default"
      };
    }
  } catch {
    // ignore: preferences are optional
  }
  return base;
}

function save(s: Saved): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    // ignore
  }
}

export const useAppearanceStore = create<AppearanceState>((set, get) => ({
  ...read(),
  init: () => {
    const s = read();
    set(s);
    apply(s);
  },
  update: (patch) => {
    set(patch);
    const { uiFont, uiFontCustom, codeFont, codeFontCustom, fontSize, replyFont } = get();
    const s = { uiFont, uiFontCustom, codeFont, codeFontCustom, fontSize, replyFont };
    save(s);
    apply(s);
  },
  reset: () => {
    const s: Saved = { ...DEFAULTS, fontSize: "default", replyFont: "serif" };
    set(s);
    save(s);
    apply(s);
  }
}));
