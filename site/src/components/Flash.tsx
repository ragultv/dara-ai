import data from "../data/benchmarks.json";
import { categoryMean, type Row } from "../lib/stats.ts";

const times = (x: number | null) => (x == null ? "n/a" : `${x.toFixed(1)}×`);

export function Flash() {
  const rows = (data.nano?.rows ?? []) as Row[];
  const edit = categoryMean(rows, "edit", "flash");
  const transform = categoryMean(rows, "transform", "flash");
  const generate = categoryMean(rows, "generate", "flash");

  return (
    <div className="sec" id="flash">
      <div className="col">
        <h2>How Flash works</h2>
        <p>
          Writing one word normally means reading the whole model from memory, and on a laptop GPU memory speed is the limit. Flash lets the model check several guessed words in a single pass, so
          the same memory read can produce more than one word.
        </p>
        <ul className="list">
          <li>
            <b>Two sources guess the next few words.</b> Prompt lookup copies from your own text when the answer repeats it, as in editing code or adding a JSON field. A small trained drafter (37 MB
            for Nano) predicts the next words from the model's own hidden state.
          </li>
          <li>
            <b>The model checks all of them at once.</b> The check costs roughly 1.5 to 2.3 times one normal step, and the model keeps the longest run it agrees with. Everything it keeps is its own
            choice.
          </li>
          <li>
            <b>It stands down when it does not pay.</b> If the guesses keep being wrong, as they often are in open writing over a long context, Flash turns itself off for that conversation
            instead of slowing you down.
          </li>
        </ul>
        <p style={{ marginTop: "var(--space-8)" }}>
          With the v3 drafter, Dara Nano is <strong className="num">{times(edit)}</strong> llama.cpp on edits, <strong className="num">{times(transform)}</strong> on rewrites and conversions, and{" "}
          <strong className="num">{times(generate)}</strong> on writing from scratch. The v3 drafter predicted 69% of the next words correctly in held-out tests (the v2 drafter, about 55%).
        </p>
      </div>
      <div className="col">
        <div className="cards" style={{ gridTemplateColumns: "1fr" }}>
          <div className="card">
            <h3>What it cannot do</h3>
            <p>
              Ordinary writing is limited by how fast the GPU reads memory, and guesses are right only part of the time, so open-ended prose gains around 1.3 to 1.6×, not several times. In our
              long-context test Flash paused on its own from about 4,000 tokens of context on a writing task.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
