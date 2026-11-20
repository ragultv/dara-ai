import React, { useId, useMemo } from "react";

export interface ChartProps {
  /** Short caption above the value. */
  label: string;
  /** Big number shown above the line (already formatted). */
  value: string;
  /** Muted text next to the caption (units, "last average"). */
  hint?: string | undefined;
  /** Newest value last. `null` is a gap (idle or not reported), never drawn as zero. */
  points: ReadonlyArray<number | null>;
  /** Number of x slots; the line fills from the right so it grows leftwards like a hardware monitor. */
  slots?: number | undefined;
  /** Fixed scale. Without them the scale follows the visible data (min 0). */
  min?: number | undefined;
  max?: number | undefined;
  /** Accessible description of the chart. */
  ariaLabel: string;
  /** Text shown when no point exists yet. */
  emptyText?: string | undefined;
  className?: string | undefined;
}

const WIDTH = 240;
const HEIGHT = 56;
const PAD = 3;

interface Run {
  xs: number[];
  ys: number[];
}

/** Pure geometry: splits the series into runs separated by gaps and maps them into the viewbox. */
export function buildRuns(
  points: ReadonlyArray<number | null>,
  slots: number,
  min: number,
  max: number
): Run[] {
  const range = max - min || 1;
  const offset = Math.max(0, slots - points.length);
  const runs: Run[] = [];
  let current: Run | null = null;
  points.forEach((p, i) => {
    if (p === null || !Number.isFinite(p)) {
      current = null;
      return;
    }
    const slot = i + offset;
    const x = PAD + (slot / Math.max(1, slots - 1)) * (WIDTH - PAD * 2);
    const clamped = Math.min(max, Math.max(min, p));
    const y = HEIGHT - PAD - ((clamped - min) / range) * (HEIGHT - PAD * 2);
    if (current === null) {
      current = { xs: [], ys: [] };
      runs.push(current);
    }
    current.xs.push(x);
    current.ys.push(y);
  });
  return runs;
}

function runToPath(run: Run): string {
  return run.xs
    .map((x, i) => `${i === 0 ? "M" : "L"} ${x.toFixed(2)} ${(run.ys[i] ?? 0).toFixed(2)}`)
    .join(" ");
}

export const Chart: React.FC<ChartProps> = ({
  label,
  value,
  hint,
  points,
  slots = 120,
  min,
  max,
  ariaLabel,
  emptyText = "Waiting for data",
  className = ""
}) => {
  const gradientId = useId();

  const geometry = useMemo(() => {
    const real = points.filter((p): p is number => p !== null && Number.isFinite(p));
    if (real.length === 0) return null;
    const lo = min ?? 0;
    const hi = max ?? Math.max(...real, lo + 1);
    const runs = buildRuns(points, slots, lo, hi);
    return runs.map((run) => {
      const line = runToPath(run);
      const firstX = run.xs[0] ?? 0;
      const lastX = run.xs[run.xs.length - 1] ?? 0;
      const isDot = run.xs.length === 1;
      return {
        line,
        area: isDot
          ? ""
          : `${line} L ${lastX.toFixed(2)} ${HEIGHT} L ${firstX.toFixed(2)} ${HEIGHT} Z`,
        isDot,
        x: firstX,
        y: run.ys[0] ?? 0
      };
    });
  }, [points, slots, min, max]);

  return (
    <div className={`rounded-2xl border border-border bg-surface p-4 text-left ${className}`}>
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-text-muted">
          {label}
        </span>
        {hint && <span className="truncate text-[11px] text-text-muted">{hint}</span>}
      </div>
      <div className="mt-1 text-2xl font-bold tabular-nums tracking-tight text-text">{value}</div>

      <div className="mt-2 h-14 w-full rounded-lg bg-surface-sunken">
        {geometry ? (
          <svg
            role="img"
            aria-label={ariaLabel}
            viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
            preserveAspectRatio="none"
            className="h-full w-full overflow-visible"
          >
            <defs>
              <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="0%"
                  style={{ stopColor: "var(--color-primary)", stopOpacity: 0.35 }}
                />
                <stop offset="100%" style={{ stopColor: "var(--color-primary)", stopOpacity: 0 }} />
              </linearGradient>
            </defs>
            {geometry.map((g, i) =>
              g.isDot ? (
                <path
                  key={i}
                  d={`M ${g.x} ${g.y} h 0.01`}
                  fill="none"
                  strokeWidth={5}
                  strokeLinecap="round"
                  vectorEffect="non-scaling-stroke"
                  style={{ stroke: "var(--color-primary)" }}
                />
              ) : (
                <g key={i}>
                  <path d={g.area} fill={`url(#${gradientId})`} />
                  <path
                    d={g.line}
                    fill="none"
                    strokeWidth={2}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    vectorEffect="non-scaling-stroke"
                    style={{ stroke: "var(--color-primary)" }}
                  />
                </g>
              )
            )}
          </svg>
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-text-muted">
            {emptyText}
          </div>
        )}
      </div>
    </div>
  );
};
