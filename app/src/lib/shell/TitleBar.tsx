import React, { useEffect, useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { Logo } from "../ui/Logo.tsx";

/** True inside the Tauri window (the browser dev view has the browser's own chrome). */
export const isTauriWindow = (): boolean =>
  typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

const glyph = {
  minimize: <path d="M1 5.5h8" />,
  maximize: <rect x="1.5" y="1.5" width="7" height="7" />,
  restore: (
    <>
      <rect x="1.5" y="3" width="5.5" height="5.5" />
      <path d="M3.5 3V1.5h5v5H7" />
    </>
  ),
  close: <path d="M1.5 1.5l7 7M8.5 1.5l-7 7" />
};

const ControlButton: React.FC<{
  label: string;
  onClick: () => void;
  icon: keyof typeof glyph;
  danger?: boolean;
}> = ({ label, onClick, icon, danger }) => (
  <button
    type="button"
    aria-label={label}
    title={label}
    onClick={onClick}
    tabIndex={-1}
    className={`h-full w-14 inline-flex items-center justify-center text-text-muted transition-colors cursor-default ${
      danger ? "hover:bg-danger hover:text-white" : "hover:bg-surface-raised hover:text-text"
    }`}
  >
    <svg
      width="11"
      height="11"
      viewBox="0 0 10 10"
      fill="none"
      stroke="currentColor"
      strokeWidth="1"
      aria-hidden="true"
    >
      {glyph[icon]}
    </svg>
  </button>
);

/** Custom window title bar (the native one is off, see tauri.conf.json): the Dara logo only, minimize / maximize / close. */
export const TitleBar: React.FC = () => {
  const [maximized, setMaximized] = useState(false);

  useEffect(() => {
    const win = getCurrentWindow();
    let unlisten: (() => void) | undefined;
    let disposed = false;
    const sync = () => {
      win
        .isMaximized()
        .then((m) => {
          if (!disposed) setMaximized(m);
        })
        .catch(() => undefined);
    };
    sync();
    win
      .onResized(sync)
      .then((u) => {
        if (disposed) u();
        else unlisten = u;
      })
      .catch(() => undefined);
    return () => {
      disposed = true;
      unlisten?.();
    };
  }, []);

  const win = getCurrentWindow();

  return (
    <header
      data-tauri-drag-region
      className="relative flex h-12 shrink-0 items-center justify-between border-b border-border bg-bg-subtle text-xs select-none"
    >
      <div data-tauri-drag-region className="flex h-full items-center pl-4">
        <Logo alt="Dara" className="h-8 w-8 object-contain pointer-events-none" />
      </div>

      <div className="flex h-full items-stretch">
        <ControlButton label="Minimize" icon="minimize" onClick={() => void win.minimize()} />
        <ControlButton
          label={maximized ? "Restore" : "Maximize"}
          icon={maximized ? "restore" : "maximize"}
          onClick={() => void win.toggleMaximize()}
        />
        <ControlButton label="Close" icon="close" danger onClick={() => void win.close()} />
      </div>
    </header>
  );
};
