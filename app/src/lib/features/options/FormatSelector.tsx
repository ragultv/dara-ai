import React, { useRef } from "react";
import type { FormatInfo, KvFormat } from "../../api/types.ts";
import { Badge } from "../../ui/Badge.tsx";
import { Icon } from "../../ui/Icon.tsx";
import { qualityPresentation } from "./optionsLogic.ts";

interface Props {
  formats: FormatInfo[];
  value: KvFormat;
  recommendedKv: KvFormat | null;
  disabled: boolean;
  onChange: (kv: KvFormat) => void;
}

/** K/V format picker: one radio card per format the server reports. */
export const FormatSelector: React.FC<Props> = ({
  formats,
  value,
  recommendedKv,
  disabled,
  onChange
}) => {
  const refs = useRef<Array<HTMLButtonElement | null>>([]);

  const onKeyDown = (e: React.KeyboardEvent, index: number): void => {
    const step =
      e.key === "ArrowRight" || e.key === "ArrowDown"
        ? 1
        : e.key === "ArrowLeft" || e.key === "ArrowUp"
          ? -1
          : 0;
    if (step === 0 || formats.length === 0) return;
    e.preventDefault();
    const next = (index + step + formats.length) % formats.length;
    const target = formats[next];
    if (target) {
      onChange(target.id);
      refs.current[next]?.focus();
    }
  };

  return (
    <div className="space-y-2">
      <h2 id="kv-label" className="text-sm font-bold text-text">
        Memory format for the conversation (K/V)
      </h2>
      <p className="text-xs text-text-muted">
        This is how Dara stores what it has read so far. A smaller format leaves room for a longer
        conversation but can lower quality.
      </p>
      <div
        role="radiogroup"
        aria-labelledby="kv-label"
        className="grid grid-cols-1 sm:grid-cols-2 gap-3"
      >
        {formats.map((f, i) => {
          const selected = f.id === value;
          const q = qualityPresentation(f.quality.label);
          return (
            <button
              key={f.id}
              ref={(el) => {
                refs.current[i] = el;
              }}
              type="button"
              role="radio"
              aria-checked={selected}
              tabIndex={selected ? 0 : -1}
              disabled={disabled}
              onClick={() => onChange(f.id)}
              onKeyDown={(e) => onKeyDown(e, i)}
              className={`text-left rounded-xl border p-3 space-y-1.5 cursor-pointer disabled:cursor-not-allowed disabled:opacity-60 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus ${
                selected
                  ? "border-primary bg-primary-subtle ring-1 ring-primary"
                  : "border-border bg-surface hover:bg-surface-raised"
              }`}
            >
              <span className="flex items-center justify-between gap-2">
                <span className="text-sm font-semibold text-text">{f.name}</span>
                {selected && <Icon name="check" size={14} className="text-primary shrink-0" />}
              </span>
              <span className="flex flex-wrap items-center gap-1.5">
                <Badge variant={q.variant} size="sm">
                  <Icon name={q.icon} size={10} className="mr-1" />
                  {f.quality.text}
                </Badge>
                {f.id === recommendedKv && (
                  <Badge variant="neutral" size="sm">
                    Recommended
                  </Badge>
                )}
              </span>
              <span className="block text-xs text-text-muted leading-snug">{f.description}</span>
              {f.caveat && (
                <span className="block text-xs text-text leading-snug">Note: {f.caveat}</span>
              )}
              <span className="block text-[11px] text-text-muted font-mono tabular-nums">
                {f.bits_per_value} bits per value
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
};
