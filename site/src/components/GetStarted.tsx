import { DOCS, RELEASES, REPO } from "../data/facts.ts";

export function GetStarted() {
  return (
    <div className="sec" id="start">
      <div className="col">
        <h2>Get started</h2>
        <ul className="list">
          <li>
            <b>Install.</b> Download <code>Dara_&lt;version&gt;_x64-setup.exe</code> from the releases page. It installs for your user account only, with no administrator prompt. You need an NVIDIA GPU
            from the GTX 16 or RTX 20 series on, and a driver that supports CUDA 13 (R580 or newer).
          </li>
          <li>
            <b>Pick a model.</b> Open Models in the app, choose one that fits your GPU and download it (a few GB). Models are kept when you uninstall the app.
          </li>
          <li>
            <b>Chat.</b> Ask questions, attach documents or images, or add tool connectors. Flash can be switched on or off in a conversation's settings.
          </li>
        </ul>
        <p className="caption">
          The installer is not code-signed yet, so Windows SmartScreen will warn when you run it (choose More info, then Run anyway). There is no automatic update yet: install the new version over the
          old one.
        </p>

        <h2>Use it from your own code</h2>
        <pre className="code">
          <span className="c"># download a model, start the local server</span>
          {"\n"}dara pull dara-nano{"\n"}dara serve --model dara-nano{"\n\n"}
          <span className="c"># OpenAI-style request</span>
          {"\n"}curl http://127.0.0.1:11435/v1/chat/completions \{"\n"}  -H "Content-Type: application/json" \{"\n"}  -d '{"{"}"model":"dara-nano","messages":[{"{"}"role":"user","content":"Hello"{"}"}]{"}"}'
        </pre>
        <p className="caption">
          The full API and options are in the <a href={`${DOCS}/SERVER.md`}>server guide</a>.
        </p>

        <h2>Help us check your GPU</h2>
        <p>
          Every speed on this page was measured on a laptop with a 4 GB GPU. If you have a different card, the <a href={`${DOCS}/RTX_LAPTOP_CHECK.md`}>15-minute check</a> builds Dara
          for your GPU, runs the same before-and-after test, and tells you whether anything got slower. Send us the result and we will add your GPU here.
        </p>
      </div>

      <div className="wide">
        <div className="band">
          <h2>Run it on the GPU you already own</h2>
          <p>Private by default: no account, no cloud, nothing leaves your computer.</p>
          <div className="cta">
            <a className="pill primary" href={RELEASES}>
              Download for Windows
            </a>
            <a className="pill" href={REPO}>
              Source on GitHub
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
