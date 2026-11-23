// Horizontal grouped bars, in the style of a paper figure: one row per item, up to three bars (llama.cpp, Dara, Dara + Flash), the value at the end of each bar.
export interface BarRow {
  name: string;
  llama?: number | null;
  dara?: number | null;
  flash?: number | null;
}

const KINDS = ["llama", "dara", "flash"] as const;

export function Legend() {
  return (
    <div className="legend">
      <span>
        <i className="sw-llama" />
        llama.cpp
      </span>
      <span>
        <i className="sw-dara" />
        Dara plain
      </span>
      <span>
        <i className="sw-flash" />
        Dara + Flash
      </span>
    </div>
  );
}

export function Bars({ rows, max, unit, digits = 0, ticks }: { rows: BarRow[]; max?: number; unit: string; digits?: number; ticks: number[] }) {
  const top = max ?? Math.max(...rows.flatMap((r) => [r.llama, r.dara, r.flash]).filter((v): v is number => typeof v === "number"));
  const scale = top / 0.86; // leave room at the right for the value label
  return (
    <>
      <div className="bars">
        {rows.map((r) => (
          <div className="bars-row" key={r.name}>
            <div className="name">{r.name}</div>
            <div className="stack">
              {KINDS.map((k) => {
                const v = r[k];
                if (v == null) return null;
                const pct = (v / scale) * 100;
                return (
                  <div className={`b ${k}`} key={k}>
                    <i style={{ width: `${pct}%` }} />
                    <span className="num" style={{ left: `calc(${pct}% + 6px)` }}>
                      {v.toFixed(digits)}
                      {unit}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
      <div className="axis" aria-hidden="true">
        {ticks.map((t) => (
          <span key={t}>
            {t}
            {unit}
          </span>
        ))}
      </div>
    </>
  );
}
