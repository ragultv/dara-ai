import { MODELS } from "../data/facts.ts";

export function Models() {
  return (
    <div className="sec" id="models">
      <div className="col">
        <h2>Three sizes, one app</h2>
        <p>
          Models are downloaded inside the app (they are not part of the installer) as standard GGUF files at Q4_K_M, together with their image encoder. The app checks what fits your GPU before it
          loads anything and shows it, so you do not have to guess.
        </p>
      </div>
      <div className="wide figure">
        <div className="table-wrap" style={{ marginTop: 0 }}>
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Based on</th>
                <th>Download</th>
                <th>GPU memory</th>
                <th>Flash drafter</th>
              </tr>
            </thead>
            <tbody>
              {MODELS.map((m) => (
                <tr key={m.id}>
                  <td className="t">
                    {m.name}
                    <small>{m.blurb}</small>
                  </td>
                  <td>{m.base}</td>
                  <td className="num">
                    {m.modelGb.toFixed(1)} GB
                    <small style={{ display: "block", color: "var(--color-text-subtle)" }}>+ {m.visionGb.toFixed(1)} GB image encoder</small>
                  </td>
                  <td className="num">
                    from {m.minVramGb} GB
                    <small style={{ display: "block", color: "var(--color-text-subtle)" }}>comfortable at {m.fullVramGb} GB</small>
                  </td>
                  <td>{m.drafter ?? "not yet"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="caption">
          <b>Table 3.</b> Sizes and memory come from the model catalog in the repository. On our 4 GB test laptop the engine fits all 35 layers and Nano's full 131,072-token context on the GPU. Pro 9B runs
          partly on the CPU on GPUs under 8 GB, so it is noticeably slower there.
        </p>
      </div>

      <div className="col">
        <h2>What is inside</h2>
        <ul className="list">
          <li>
            <b>A native CUDA engine.</b> C++20 and CUDA, no Python or PyTorch at runtime. It reads standard GGUF files and uses 8-bit integer kernels for the heavy matrix math, CUDA Graphs to cut
            launch overhead while decoding, and attention kernels built for long conversations.
          </li>
          <li>
            <b>It fits itself to your GPU.</b> Before loading, a planner works out how many layers and how much context fit your VRAM and which cache format to use, and the app shows you the result.
            Image input costs only about 24 MB extra on Dara Nano.
          </li>
          <li>
            <b>A desktop app that stays local.</b> Built with Tauri and React. Chats, settings and documents live in a database on your computer, and nothing is sent anywhere except the model downloads
            you start. Tool connectors (MCP) are optional.
          </li>
          <li>
            <b>An API for your own tools.</b> The same engine runs as a local server with an OpenAI-compatible API on 127.0.0.1. In a 10-case tool-calling test in the app's own format, Dara Nano wrote
            the correct call every time, matching llama.cpp.
          </li>
        </ul>
      </div>
    </div>
  );
}
