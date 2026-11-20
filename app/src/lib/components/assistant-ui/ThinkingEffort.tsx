import React, { useEffect, useRef, useState } from "react";
import { Icon } from "../../ui/Icon.tsx";
import { EFFORT_LEVELS } from "../../chat/effort.ts";
import { useEffortStore } from "../../stores/useEffortStore.ts";
import { useChatStore } from "../../stores/useChatStore.ts";

const STEPS = EFFORT_LEVELS.length;

/** The stepped slider: a pill with one dot per level and a knob. Click a dot, drag the knob, or use the arrow keys. */
const EffortSlider: React.FC<{ index: number; onChange: (i: number) => void }> = ({
  index,
  onChange
}) => {
  const track = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState(false);
  const current = EFFORT_LEVELS[index]!;
  const on = index > 0;

  const indexAt = (clientX: number): number => {
    const el = track.current;
    if (!el) return index;
    const r = el.getBoundingClientRect();
    const f = (clientX - r.left) / Math.max(1, r.width);
    return Math.round(Math.min(1, Math.max(0, f)) * (STEPS - 1));
  };
  const clamp = (i: number) => Math.min(STEPS - 1, Math.max(0, i));

  return (
    <div className="h-8 w-full rounded-full border border-border bg-surface-sunken px-4 flex items-center">
      <div
        ref={track}
        role="slider"
        tabIndex={0}
        aria-label="Thinking effort"
        aria-valuemin={0}
        aria-valuemax={STEPS - 1}
        aria-valuenow={index}
        aria-valuetext={`${current.label}. ${current.description}`}
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          setDragging(true);
          onChange(indexAt(e.clientX));
        }}
        onPointerMove={(e) => {
          if (dragging) onChange(indexAt(e.clientX));
        }}
        onPointerUp={(e) => {
          setDragging(false);
          if (e.currentTarget.hasPointerCapture(e.pointerId))
            e.currentTarget.releasePointerCapture(e.pointerId);
        }}
        onPointerCancel={() => setDragging(false)}
        onKeyDown={(e) => {
          const step =
            e.key === "ArrowRight" || e.key === "ArrowUp"
              ? 1
              : e.key === "ArrowLeft" || e.key === "ArrowDown"
                ? -1
                : 0;
          if (step) {
            e.preventDefault();
            onChange(clamp(index + step));
          } else if (e.key === "Home") {
            e.preventDefault();
            onChange(0);
          } else if (e.key === "End") {
            e.preventDefault();
            onChange(STEPS - 1);
          }
        }}
        className="relative h-full w-full cursor-pointer touch-none rounded-full focus-visible:outline-2 focus-visible:outline-focus focus-visible:outline-offset-2"
      >
        {EFFORT_LEVELS.map((l, i) => (
          <span
            key={l.id}
            aria-hidden="true"
            style={{ left: `${(i / (STEPS - 1)) * 100}%` }}
            className={`absolute top-1/2 h-1.5 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full transition-colors ${
              i <= index && on ? "bg-primary" : "bg-border-strong"
            }`}
          />
        ))}
        <span
          aria-hidden="true"
          style={{ left: `${(index / (STEPS - 1)) * 100}%` }}
          className={`absolute top-1/2 h-4.5 w-4.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-text shadow-md ring-2 ring-surface-sunken ${
            dragging ? "" : "transition-[left] duration-150 ease-out"
          }`}
        />
      </div>
    </div>
  );
};

/**
 * Thinking effort for the composer: a pill that reads "Thinking: Low". Clicking it opens a small popup with the stepped slider
 * (Off, Low, Medium, High, Extra, Max), the chosen level and what it means.
 */
export const ThinkingEffort: React.FC<{ className?: string }> = ({ className = "" }) => {
  const effort = useEffortStore((s) => s.effort);
  const setEffort = useEffortStore((s) => s.setEffort);
  const isGenerating = useChatStore((s) => s.isGenerating);
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);

  const index = Math.max(
    0,
    EFFORT_LEVELS.findIndex((l) => l.id === effort)
  );
  const current = EFFORT_LEVELS[index]!;
  const on = index > 0;

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (root.current && !root.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  useEffect(() => {
    if (isGenerating) setOpen(false);
  }, [isGenerating]);

  return (
    <div ref={root} className={`relative ${className}`}>
      <button
        type="button"
        disabled={isGenerating}
        onClick={() => setOpen(!open)}
        aria-haspopup="dialog"
        aria-expanded={open}
        title={`Thinking: ${current.label}. ${current.description}`}
        className={`h-6 px-2.5 rounded-full border transition-all flex items-center gap-1.5 text-xs font-medium select-none cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed ${
          on
            ? "bg-text/15 border-border-strong text-text shadow-xs hover:bg-text/25"
            : "bg-surface-sunken border-border text-text-subtle hover:text-text hover:bg-surface-raised"
        }`}
      >
        <Icon name="brain" size={11} />
        <span className="text-[11px] font-semibold">Thinking: {current.label}</span>
        <Icon
          name="chevron-down"
          size={10}
          className={`transition-transform duration-150 ${open ? "rotate-180" : ""}`}
        />
      </button>

      {open && (
        <div
          role="dialog"
          aria-label="Thinking effort"
          className="absolute bottom-full left-0 z-40 mb-2 w-64 rounded-lg border border-border bg-surface-raised p-3 shadow-lg"
        >
          <div className="mb-2.5 flex items-baseline justify-between">
            <p className="text-[10px] font-bold uppercase tracking-wider text-text-subtle">
              Thinking effort
            </p>
            <p className="text-xs font-semibold text-text">{current.label}</p>
          </div>

          <EffortSlider index={index} onChange={(i) => setEffort(EFFORT_LEVELS[i]!.id)} />

          <div
            className="mt-1.5 flex justify-between px-1 text-[9px] font-medium text-text-subtle"
            aria-hidden="true"
          >
            <span>Off</span>
            <span>Max</span>
          </div>

          <p className="mt-2 text-[11px] leading-snug text-text-muted">{current.description}</p>
        </div>
      )}
    </div>
  );
};
