import { useState } from "react";
import data from "../data/benchmarks.json";
import { CATEGORY_LABEL, TASK_LABEL, geomean, ratios, type Row } from "../lib/stats.ts";
import { Bars, Legend } from "./Bars.tsx";

type Tier = "nano" | "core4b";
const TIERS: { key: Tier; label: string; drafter: string }[] = [
  { key: "nano", label: "Dara Nano", drafter: "Flash drafter v3" },
  { key: "core4b", label: "Dara Core 4B", drafter: "Flash drafter v2" }
];

const times = (x: number | null, d = 2) => (x == null ? "n/a" : `${x.toFixed(d)}×`);
const ms = (s: number | null) => (s == null ? "n/a" : `${Math.round(s * 1000)} ms`);

export function Results() {
  const [tier, setTier] = useState<Tier>("nano");
  const rows = ((data[tier] as { rows: Row[] } | null)?.rows ?? []) as Row[];
  const t = TIERS.find((x) => x.key === tier)!;
  const same = rows.filter((r) => r.sameTextAsPlain).length;
  const plain = geomean(ratios(rows, "plain"));
  const flash = geomean(ratios(rows, "flash"));
  const top = Math.max(...rows.flatMap((r) => [r.llama, r.plain, r.flash]).map((v) => v ?? 0));
  const step = top > 150 ? 50 : 25;
  const ticks = Array.from({ length: Math.ceil(top / step) + 1 }, (_, i) => i * step).filter((v) => v <= top / 0.86 + step);

  return (
    <div className="sec" id="results">
      <div className="col">
        <h2>Results</h2>
        <p>
          We ran sixteen everyday tasks (edits, conversions, writing) through llama.cpp and through Dara, one after the other, three times each, and kept the median. Both read the same model file and
          the same prompt, and both decode greedily. Each task is measured on Dara plain and on Dara with Flash, a small trained drafter that guesses ahead while the model checks every guess.
        </p>
      </div>

      <div className="wide figure">
        <div className="toolbar">
          <div className="tabs" role="tablist" aria-label="Model">
            {TIERS.map((x) => (
              <button key={x.key} role="tab" aria-selected={tier === x.key} onClick={() => setTier(x.key)}>
                {x.label}
              </button>
            ))}
          </div>
          <Legend />
        </div>
        {rows.length > 0 ? (
          <>
            <Bars rows={rows.map((r) => ({ name: TASK_LABEL[r.task] ?? r.task, llama: r.llama, dara: r.plain, flash: r.flash }))} unit="" digits={0} ticks={ticks} />
            <p className="caption">
              <b>Figure 1.</b> Decode speed in tokens per second while writing the reply, {t.label} with {t.drafter} on a 4 GB laptop GPU. Dara plain is {times(plain)} and Dara with Flash {times(flash)} llama.cpp on average. Edits and conversions gain the
              most because most of the answer is already in the prompt; writing from scratch gains less.
            </p>
          </>
        ) : (
          <p className="note">The {t.label} numbers are being re-measured on the current engine and will appear here.</p>
        )}
      </div>

      <div className="col">
        <p style={{ marginTop: "var(--space-10)" }}>
          With Flash off, plain decoding is {times(plain)} llama.cpp on {t.label}: the model has to read all of its weights for every word and both engines read them at nearly the same speed. The
          bigger gains come from getting more than one word out of each read.
        </p>
      </div>

      <div className="wide figure">
        <div className="table-wrap" style={{ marginTop: 0 }}>
          <table>
            <thead>
              <tr>
                <th>Prompt</th>
                <th>llama.cpp tok/s</th>
                <th>Dara plain tok/s</th>
                <th>Dara + Flash tok/s</th>
                <th>Flash vs llama.cpp</th>
                <th>llama.cpp first word</th>
                <th>Dara first word</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const x = r.llama && r.flash ? r.flash / r.llama : null;
                return (
                  <tr key={r.task}>
                    <td className="t">
                      {TASK_LABEL[r.task] ?? r.task}
                      <small>
                        {CATEGORY_LABEL[r.category]?.toLowerCase()}, {r.promptTokens} tokens in
                      </small>
                    </td>
                    <td className="num">{r.llama?.toFixed(1)}</td>
                    <td className="num">{r.plain?.toFixed(1)}</td>
                    <td className="num hl">{r.flash?.toFixed(1)}</td>
                    <td className="num">
                      <b>{times(x)}</b>
                    </td>
                    <td className="num">{ms(r.ttftLlama)}</td>
                    <td className="num">{ms(r.ttftDara)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="caption">
          <b>Table 1.</b> The sixteen prompts for {t.label}. Medians of three runs, up to 256 new tokens, llama.cpp build b12189 with flash attention, identical GGUF files (Q4_K_M), a laptop with a 4 GB GPU on AC
          power. "First word" is the time from sending the prompt to the first word of the reply. Reproduce it with <code>bench/husky_bench.py</code>.
        </p>
      </div>
    </div>
  );
}
