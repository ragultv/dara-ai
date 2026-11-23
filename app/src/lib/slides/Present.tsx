import React, { useCallback, useEffect, useRef, useState } from "react";
import { Icon } from "../ui/Icon.tsx";
import { SlideSurface } from "./SlideSurface.tsx";
import type { Deck } from "./deck.ts";

const isDesktop = () => typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

/** Full screen for the window (the desktop app) or the page (a browser). Returns a way back; failure is not an error, the slide still fills the window. */
async function enterFullscreen(): Promise<() => Promise<void>> {
  try {
    if (isDesktop()) {
      const { getCurrentWindow } = await import("@tauri-apps/api/window");
      const win = getCurrentWindow();
      await win.setFullscreen(true);
      return () => win.setFullscreen(false);
    }
    await document.documentElement.requestFullscreen();
    return async () => {
      if (document.fullscreenElement) await document.exitFullscreen();
    };
  } catch {
    return async () => {};
  }
}

interface Props {
  deck: Deck;
  docId: string;
  title: string;
  /** the slide to start on */
  start: number;
  onExit: () => void;
}

/**
 * Presenting: one slide fills the screen. Right arrow, Space, Enter, PageDown or a click go forward; left arrow, PageUp or Backspace
 * go back; Home and End jump; Escape ends. The controls show for a moment when the mouse moves.
 */
export const Present: React.FC<Props> = ({ deck, docId, title, start, onExit }) => {
  const last = deck.slides.length - 1;
  const [index, setIndex] = useState(Math.min(Math.max(0, start), last));
  const [chrome, setChrome] = useState(true);
  const hideTimer = useRef<number | undefined>(undefined);
  const root = useRef<HTMLDivElement>(null);
  const leave = useRef<(() => Promise<void>) | null>(null);

  const go = useCallback((to: number) => setIndex(Math.min(Math.max(0, to), last)), [last]);

  const exit = useCallback(() => {
    const back = leave.current;
    leave.current = null;
    void (back ? back() : Promise.resolve()).finally(onExit);
  }, [onExit]);

  // Full screen is entered once per presentation. React runs an effect twice in development (mount, clean up, mount): the first run's
  // cleanup must not leave the full screen the second run depends on, so "still here" is a ref, and the request is made only once.
  const mounted = useRef(false);
  const started = useRef(false);
  useEffect(() => {
    mounted.current = true;
    if (!started.current) {
      started.current = true;
      void enterFullscreen().then((back) => {
        if (!mounted.current) void back();
        else leave.current = back;
      });
    }
    root.current?.focus();
    return () => {
      mounted.current = false;
      const back = leave.current;
      leave.current = null;
      if (back) void back();
    };
  }, []);

  // Pressing Escape in a browser's full screen ends the full screen, not the presentation: follow it. Only a full screen this
  // presentation was in counts, so a late "left full screen" from a presentation that just ended cannot end this one.
  const entered = useRef(false);
  useEffect(() => {
    if (isDesktop()) return;
    const onChange = () => {
      if (document.fullscreenElement) entered.current = true;
      else if (entered.current) exit();
    };
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, [exit]);

  const poke = () => {
    setChrome(true);
    window.clearTimeout(hideTimer.current);
    hideTimer.current = window.setTimeout(() => setChrome(false), 2200);
  };
  useEffect(() => {
    poke();
    return () => window.clearTimeout(hideTimer.current);
  }, []);

  const onKey = (e: React.KeyboardEvent) => {
    switch (e.key) {
      case "ArrowRight":
      case "ArrowDown":
      case "PageDown":
      case " ":
      case "Enter":
        go(index + 1);
        break;
      case "ArrowLeft":
      case "ArrowUp":
      case "PageUp":
      case "Backspace":
        go(index - 1);
        break;
      case "Home":
        go(0);
        break;
      case "End":
        go(last);
        break;
      case "Escape":
        exit();
        break;
      default:
        return;
    }
    e.preventDefault();
    poke();
  };

  const slide = deck.slides[index] ?? "";
  return (
    <div
      ref={root}
      className="slide-present outline-none"
      role="dialog"
      aria-modal="true"
      aria-label={`Presenting ${title || "slides"}`}
      tabIndex={-1}
      onKeyDown={onKey}
      onMouseMove={poke}
      onClick={(e) => {
        // the left third goes back, the rest goes forward; the controls handle their own clicks
        if ((e.target as HTMLElement).closest("[data-present-controls]")) return;
        const x = e.clientX / window.innerWidth;
        go(x < 0.33 ? index - 1 : index + 1);
      }}
      style={{ cursor: chrome ? "default" : "none" }}
    >
      <SlideSurface key={index} markdown={slide} theme={deck.theme} docId={docId} />
      <div
        aria-hidden="true"
        className="absolute inset-x-0 bottom-0 h-1 bg-white/10"
        style={{ opacity: chrome ? 1 : 0.35 }}
      >
        <div
          className="h-full bg-white/70 transition-[width] duration-200"
          style={{ width: `${((index + 1) / deck.slides.length) * 100}%` }}
        />
      </div>
      <div
        data-present-controls
        className={`absolute bottom-5 left-1/2 flex -translate-x-1/2 items-center gap-1 rounded-full bg-black/70 px-2 py-1.5 text-white shadow-lg transition-opacity duration-300 ${
          chrome ? "opacity-100" : "pointer-events-none opacity-0"
        }`}
      >
        <button
          type="button"
          aria-label="Previous slide"
          disabled={index === 0}
          onClick={() => go(index - 1)}
          className="flex size-8 cursor-pointer items-center justify-center rounded-full hover:bg-white/15 disabled:cursor-default disabled:opacity-40"
        >
          <Icon name="arrow-left" size={16} />
        </button>
        <span className="min-w-14 text-center text-xs tabular-nums" role="status">
          {index + 1} / {deck.slides.length}
        </span>
        <button
          type="button"
          aria-label="Next slide"
          disabled={index === last}
          onClick={() => go(index + 1)}
          className="flex size-8 cursor-pointer items-center justify-center rounded-full hover:bg-white/15 disabled:cursor-default disabled:opacity-40"
        >
          <Icon name="arrow-right" size={16} />
        </button>
        <span aria-hidden="true" className="mx-1 h-4 w-px bg-white/25" />
        <button
          type="button"
          aria-label="End presentation"
          title="End (Esc)"
          onClick={exit}
          className="flex size-8 cursor-pointer items-center justify-center rounded-full hover:bg-white/15"
        >
          <Icon name="x" size={16} />
        </button>
      </div>
    </div>
  );
};
