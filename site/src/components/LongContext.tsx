import data from "../data/benchmarks.json";

interface Point {
  tokens: number;
  plainDecode: number | null;
  flashDecode: number | null;
  coldFirstTokenSeconds: number | null;
  prefillTps: number | null;
}
const W = 760;
const H = 300;
const M = { l: 44, r: 24, t: 12, b: 40 };
const X_MIN = 50;
const X_MAX = 90000;
const Y_MAX = 80;
const x = (t: number) => M.l + ((Math.log10(Math.max(t, X_MIN)) - Math.log10(X_MIN)) / (Math.log10(X_MAX) - Math.log10(X_MIN))) * (W - M.l - M.r);
const y = (v: number) => H - M.b - (v / Y_MAX) * (H - M.t - M.b);
const path = (pts: Point[]) => pts.filter((p) => p.plainDecode != null).map((p, i) => `${i ? "L" : "M"}${x(p.tokens).toFixed(1)},${y(p.plainDecode as number).toFixed(1)}`).join(" ");

const kfmt = (t: number) => (t >= 1000 ? `${Math.round(t / 1000)}K` : `${t}`);
const seconds = (s: number | null) => (s == null ? "n/a" : s < 10 ? `${s.toFixed(1)} s` : s < 120 ? `${Math.round(s)} s` : `${(s / 60).toFixed(1)} min`);

export function LongContext() {
  const after = (data.contextAfter ?? []) as Point[];
  if (!after.length) return null;
  const last = after[after.length - 1];
  const ticksX = [100, 1000, 4000, 16000, 64000];
  const ticksY = [0, 20, 40, 60, 80];

  return (
    <div className="sec" id="context">
      <div className="col">
        <h2>Long conversations stay usable</h2>
        <p>
          Every new word has to look back over the whole conversation, so decoding slows as the conversation grows. Dara reads that history once per group of attention heads instead of once per
          head, which cuts the memory it moves and keeps the speed up much further out.
        </p>
      </div>

      <div className="wide figure">
        <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-labelledby="ctx-title ctx-desc" style={{ marginTop: "var(--space-4)" }}>
          <title id="ctx-title">Decode speed against context length, Dara Nano</title>
          <desc id="ctx-desc">
            At {kfmt(last.tokens)} tokens of context decoding runs at {last.plainDecode} tokens per second.
          </desc>
          {ticksY.map((v) => (
            <g key={v}>
              <line x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} stroke="var(--color-border)" />
              <text x={M.l - 8} y={y(v) + 4} textAnchor="end">
                {v}
              </text>
            </g>
          ))}
          {ticksX.map((t) => (
            <text key={t} x={x(t)} y={H - M.b + 18} textAnchor="middle">
              {kfmt(t)}
            </text>
          ))}
          <text x={W / 2} y={H - 4} textAnchor="middle">
            tokens of context (log scale) · tokens per second
          </text>
                    <path d={path(after)} fill="none" stroke="var(--color-primary)" strokeWidth="2.5" />
          {after.filter((p) => p.plainDecode != null).map((p) => (
            <circle key={p.tokens} cx={x(p.tokens)} cy={y(p.plainDecode as number)} r="3.5" fill="var(--color-primary)" />
          ))}
        </svg>
        <p className="caption">
          <b>Figure 2.</b> Plain decoding (no Flash), Dara Nano on a 4 GB laptop GPU, measured up to {kfmt(last.tokens)} tokens. At {kfmt(last.tokens)} it still decodes at {last.plainDecode}{" "}
          tokens per second. The model supports 131,072 tokens; we did not benchmark beyond what is shown.
        </p>
      </div>

      <div className="wide figure">
        <div className="table-wrap" style={{ marginTop: 0 }}>
          <table>
            <thead>
              <tr>
                <th>Context</th>
                <th>Decode</th>
                <th>Reading the prompt</th>
                <th>First word, new prompt</th>
              </tr>
            </thead>
            <tbody>
              {after.map((p) => (
                <tr key={p.tokens}>
                  <td className="num t">{kfmt(p.tokens)} tokens</td>
                  <td className="num hl">
                    <b>{p.plainDecode ?? "n/a"}</b>
                  </td>
                  <td className="num">{p.prefillTps ? `${p.prefillTps} tok/s` : "n/a"}</td>
                  <td className="num">{seconds(p.coldFirstTokenSeconds)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="caption">
          <b>Table 2.</b> Decode speed in tokens per second. "First word" is the time to the first word of the reply for a brand-new prompt of that length. Reading a long prompt for the first time is real
          work on a laptop GPU and grows with length; no setting removes it.
        </p>
      </div>

      <div className="col">
        <p style={{ marginTop: "var(--space-10)" }}>
          <strong>Coming back to a conversation is quick.</strong> Dara keeps recent conversations' state in your computer's RAM. Returning to a chat you already had skips the long read: on Dara Nano
          the first word came in about 0.15 s instead of 1.7 to 2.1 s, and on Core 4B in 0.4 s instead of 3.3 to 3.8 s (conversations of about 1,300 tokens). A conversation switched back to after a restart is read again.
        </p>
      </div>

    </div>
  );
}
