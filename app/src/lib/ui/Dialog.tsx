import React, { useEffect, useRef } from "react";
import { Icon } from "./Icon.tsx";

export interface DialogProps {
  isOpen: boolean;
  title: string;
  description?: string;
  onclose: () => void;
  children?: React.ReactNode;
  actions?: React.ReactNode;
  /** what gets focus when the dialog opens (default: the first focusable element) */
  initialFocus?: React.RefObject<HTMLElement | null>;
}

export const Dialog: React.FC<DialogProps> = ({
  isOpen,
  title,
  description,
  onclose,
  children,
  actions,
  initialFocus
}) => {
  const dialogRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onclose();
      }
    };

    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", handleKeyDown);

    const timer = setTimeout(() => {
      if (initialFocus?.current) {
        initialFocus.current.focus();
        return;
      }
      const focusable = dialogRef.current?.querySelectorAll<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      if (focusable && focusable.length > 0) {
        focusable[0]?.focus();
      }
    }, 50);

    return () => {
      document.body.style.overflow = "";
      window.removeEventListener("keydown", handleKeyDown);
      clearTimeout(timer);
    };
  }, [isOpen, onclose, initialFocus]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/60 backdrop-blur-xs transition-opacity animate-in fade-in duration-200"
        onClick={onclose}
        aria-hidden="true"
      />

      {/* Modal Dialog */}
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className="relative w-full max-w-lg rounded-2xl bg-surface-raised border border-border p-6 shadow-2xl z-10 text-left animate-in zoom-in-95 fade-in duration-200 space-y-4"
      >
        <div className="flex items-start justify-between gap-4">
          <div className="space-y-1">
            <h2 className="text-base font-bold text-text leading-tight">{title}</h2>
            {description && (
              <p className="text-xs text-text-muted leading-relaxed">{description}</p>
            )}
          </div>

          <button
            type="button"
            onClick={onclose}
            aria-label="Close dialog"
            className="p-1 rounded-lg text-text-subtle hover:text-text hover:bg-surface transition-colors cursor-pointer"
          >
            <Icon name="x" size={16} />
          </button>
        </div>

        {children && <div className="text-sm text-text-muted space-y-2">{children}</div>}

        {actions && (
          <div className="flex items-center justify-end gap-2 pt-4 border-t border-border">
            {actions}
          </div>
        )}
      </div>
    </div>
  );
};
