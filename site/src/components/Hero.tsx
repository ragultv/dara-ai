import data from "../data/benchmarks.json";
import { RELEASES, REPO } from "../data/facts.ts";
import { CATEGORY_LABEL, categoryMean, geomean, ratios, TASK_LABEL, type Row } from "../lib/stats.ts";
import { Bars, Legend } from "./Bars.tsx";

export function Hero() {
  const rows = (data.nano?.rows ?? []) as Row[];
  const flash = geomean(ratios(rows, "flash"));
  const plain = geomean(ratios(rows, "plain"));
  const edit = categoryMean(rows, "edit", "flash");
  const ttft = rows.filter((r) => r.ttftLlama && r.ttftDara).map((r) => (r.ttftLlama as number) / (r.ttftDara as number));
  const demo = rows.find((r) => r.task === "function_edit") ?? rows[0];

  const cats = (["edit", "transform", "generate"] as const).map((c) => ({
    name: CATEGORY_LABEL[c],
    llama: 1,
    dara: categoryMean(rows, c, "plain"),
    flash: categoryMean(rows, c, "flash")
  }));

  return (
    <>
      <section className="hero" id="top">
        <div className="wide">
          <div className="hero-card">
            <div className="hero-top">
              <span className="brand">
                <img src="/dara-light.png" alt="" />
                <span>Dara</span>
              </span>
              <span className="label">Windows · NVIDIA laptop GPUs</span>
            </div>
            <h1>
              Most local AI runs every model the same way. <strong>Dara is built for the laptop GPU you already have.</strong>
            </h1>
            {demo && (
              <div className="hero-race" role="img" aria-label={`${TASK_LABEL[demo.task]}: llama.cpp ${demo.llama} tokens per second, Dara plain ${demo.plain}, Dara with Flash ${demo.flash}`}>
                <div className="race">
                  {(
                    [
                      ["llama.cpp", demo.llama, ""],
                      ["Dara plain", demo.plain, "dara"],
                      ["Dara + Flash", demo.flash, "flash"]
                    ] as const
                  ).map(([who, v, cls]) => (
                    <div className={`race-row ${cls}`} key={who}>
                      <div className="who">{who}</div>
                      <div className="track">
                        <div className="fill num" style={{ width: `${((v ?? 0) / (demo.flash ?? 1)) * 100}%` }}>
                          {v?.toFixed(0)} tok/s
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
                <p className="caption">
                  <b>Editing a function</b>, Dara Nano on a 4 GB laptop GPU. Same model file, same prompt.
                </p>
              </div>
            )}
            <div className="cta">
              <a className="pill primary" href={RELEASES}>
                Download for Windows
              </a>
              <a className="pill" href={REPO}>
                View on GitHub
              </a>
            </div>
          </div>
        </div>
      </section>

      <div className="col article">
        <p className="label">Engine · Local AI · Dara</p>
        <h1 className="title">Dara: a local engine for laptop GPUs, {flash ? `${flash.toFixed(1)}×` : ""} faster than llama.cpp</h1>
        <p className="dek">
          Dara is a Windows app with its own CUDA engine for Gemma 4 and Qwen3.5 models. Chat, documents, images and tool calls run on your own graphics card, and your conversations stay in a
          database on your computer. It is built for laptop GPUs with 4 to 12 GB of memory.
        </p>
        <div className="byline">
          <span>
            <b>Measured</b> {data.generated}
          </span>
          <span>4 GB VRAM laptop GPU</span>
          <span>against llama.cpp b12189</span>
        </div>
      </div>

      <div className="wide">
        <div className="panel">
          <div className="stats">
            <div className="stat">
              <div className="big num">{flash ? `${flash.toFixed(2)}×` : "n/a"}</div>
              <div className="what">with Flash, against llama.cpp</div>
              <div className="sub">
                Dara Nano with Flash, average over {rows.length} tasks on 4 GB of VRAM. Edits: {edit ? `${edit.toFixed(2)}×` : "n/a"}.
              </div>
            </div>
            <div className="stat">
              <div className="big num">{plain ? `${plain.toFixed(2)}×` : "n/a"}</div>
              <div className="what">plain decoding, no Flash</div>
              <div className="sub">Same model file as llama.cpp, before any guessing ahead.</div>
            </div>
            <div className="stat">
              <div className="big num">{ttft.length ? `${Math.min(...ttft).toFixed(1)}–${Math.max(...ttft).toFixed(1)}×` : "n/a"}</div>
              <div className="what">sooner to the first word</div>
              <div className="sub">New prompts of 120 to 860 tokens, Dara Nano against llama.cpp.</div>
            </div>
            <div className="stat">
              <div className="big num">131K</div>
              <div className="what">tokens of context on 4 GB</div>
              <div className="sub">Dara Nano keeps its full context window on the GPU.</div>
            </div>
          </div>
          <div className="panel-fig">
            <h3>Flash helps most where the answer already sits in your prompt</h3>
            <p className="sub">Speed relative to llama.cpp (1×), by kind of task. Dara Nano, 4 GB laptop GPU.</p>
            <div style={{ marginTop: "var(--space-5)" }}>
              <Legend />
            </div>
            <Bars rows={cats} max={3} unit="×" digits={2} ticks={[0, 1, 2, 3]} />
          </div>
        </div>
      </div>
    </>
  );
}
