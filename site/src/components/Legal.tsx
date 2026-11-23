import type { ReactNode } from "react";
import { REPO } from "../data/facts.ts";
import { Footer } from "./Footer.tsx";
import { Navbar } from "./Navbar.tsx";

export const UPDATED = "23 November 2026";
const ISSUES = `${REPO}/issues`;

/** The frame of a legal page: the same header and footer as the home page, a title, and the reading column. */
export function LegalPage({ title, intro, children }: { title: string; intro: string; children: ReactNode }) {
  return (
    <>
      <Navbar base="/" />
      <main id="main" className="article">
        <div className="col">
          <p className="label">Legal</p>
          <h1 className="title">{title}</h1>
          <p className="dek">{intro}</p>
          <div className="byline">
            <span>
              <b>Last updated</b> {UPDATED}
            </span>
          </div>
        </div>
        <div className="sec" style={{ paddingTop: "var(--space-8)" }}>
          <div className="col legal">{children}</div>
        </div>
      </main>
      <Footer />
    </>
  );
}

export function Privacy() {
  return (
    <LegalPage
      title="Privacy"
      intro="Dara runs on your computer. This page says exactly what stays there, and the few things that do cross the network."
    >
      <h2>The short version</h2>
      <p>
        Dara has no accounts, no analytics, no crash reporting and no update checks. Your chats, documents and images are processed by the engine on your own graphics card and stored in a database on your
        own disk. We do not receive them, and we have no server that could.
      </p>

      <h2>What stays on your computer</h2>
      <ul className="list">
        <li>
          <b>Conversations, settings, documents and attached images.</b> They are stored in a local database and attachments folder under <code>%LOCALAPPDATA%\Dara</code>. Deleting them in the app
          removes the files and the database entries.
        </li>
        <li>
          <b>Your prompts and the model's answers.</b> They are handled by the engine on your machine. The engine listens on <code>127.0.0.1</code> only, and every request needs a random key created
          each time the app starts. If you turn on LAN access in Settings, the app warns you and requires the key.
        </li>
        <li>
          <b>Models.</b> They are stored in the models folder you choose (by default <code>%LOCALAPPDATA%\Dara\models</code>) and are kept when you uninstall the app.
        </li>
      </ul>

      <h2>What crosses the network</h2>
      <p>Only these, and only when you start them:</p>
      <ul className="list">
        <li>
          <b>Model downloads.</b> When you download a model in the app, the engine fetches the model files from Hugging Face and the Flash drafter files from this project's GitHub releases. Those services
          see your IP address and the files you request, under their own privacy policies. Each file is checked against a known SHA-256 hash.
        </li>
        <li>
          <b>The installer.</b> If Microsoft's WebView2 runtime is missing, the installer downloads it from Microsoft.
        </li>
        <li>
          <b>Tool connectors you add.</b> Connectors (MCP) are optional and off until you add one. A connector you add talks to the service it belongs to (for example a calendar or mail service), and what
          it sends there is what you ask it to send, under that service's policy. The app asks before a tool runs.
        </li>
      </ul>

      <h2>What we do not do</h2>
      <ul className="list">
        <li>We do not collect usage data, device identifiers or crash reports.</li>
        <li>We do not log your messages or your API key. Logs hold events and error codes only.</li>
        <li>We do not sell or share data, because we do not have any.</li>
        <li>The app loads no remote scripts, fonts or images: its content security policy blocks them, so a model's reply cannot make the app fetch an image from the internet.</li>
      </ul>

      <h2>This website</h2>
      <p>
        This site sets no cookies, runs no analytics and loads no third-party scripts, fonts or trackers. Whoever hosts it, like any web host, may keep ordinary server logs (IP address, time, page requested).
        Links to GitHub, Hugging Face and others take you to sites with their own policies.
      </p>

      <h2>Your control</h2>
      <p>
        Because the data is on your computer, you control it: delete conversations in the app, delete the folders above, or uninstall. Dara is open source, so you can read in the repository exactly what the
        app does, including everything on this page.
      </p>

      <h2>Changes and questions</h2>
      <p>
        If this page changes, the date above changes with it, and the history is in the repository. Questions or concerns: <a href={ISSUES}>open an issue on GitHub</a>.
      </p>
    </LegalPage>
  );
}

export function Terms() {
  return (
    <LegalPage
      title="Terms"
      intro="Dara is free, open-source software that runs on your own computer. These terms say what you can do with it and what we do not promise."
    >
      <h2>The software</h2>
      <p>
        Dara (the desktop app, the engine and the code in the repository) is released under the <a href={`${REPO}/blob/main/LICENSE`}>MIT License</a>. You may use, copy, modify, merge, publish, distribute,
        sublicense and sell it, as long as you keep the copyright and license notice. Third-party code in <code>third_party/</code> keeps its own license.
      </p>

      <h2>No warranty</h2>
      <p>
        The software is provided "as is", without warranty of any kind, express or implied, including merchantability, fitness for a particular purpose and non-infringement. This is an early release:
        it may crash, lose conversation data, or not work on your hardware. Keep a copy of anything you cannot afford to lose. To the extent the law allows, the authors are not liable for any claim or
        damages arising from using it.
      </p>

      <h2>Models are not ours</h2>
      <p>
        Dara downloads language models (for example Gemma and Qwen) from their publishers. Those models are not part of Dara and are covered by their own licenses and use policies, which you accept when
        you download and use them. You are responsible for following them. The Flash drafter files are trained to work with specific models and may carry those models' terms.
      </p>

      <h2>What the model says</h2>
      <p>
        Language models make mistakes and can produce text that is wrong, biased or inappropriate. Check anything that matters before you rely on it. Do not use Dara's output as medical, legal, financial or
        safety advice. You are responsible for what you ask for, what you do with the answers, and for the documents and images you give it.
      </p>

      <h2>Acceptable use</h2>
      <p>
        Do not use Dara to break the law, to infringe other people's rights, or to harm others. If you add tool connectors, you are responsible for the accounts and data they can reach, and for approving
        what a tool does when the app asks.
      </p>

      <h2>Speed and benchmark numbers</h2>
      <p>
        Speeds on this site and in the repository were measured on specific laptops with specific settings, and are described with how they were measured. Your results will differ with your GPU, driver
        and settings.
      </p>

      <h2>Name and logo</h2>
      <p>
        The MIT License covers the code. It does not give you the right to use the Dara name or logo to suggest that your fork or product is the official one. Please name your own project differently.
      </p>

      <h2>This website</h2>
      <p>The website is provided for information. We may change or remove it at any time.</p>

      <h2>Changes and questions</h2>
      <p>
        We may update these terms; the date above shows the latest version and the history is in the repository. Questions: <a href={ISSUES}>open an issue on GitHub</a>. See also the <a href="/privacy.html">Privacy</a> page.
      </p>
    </LegalPage>
  );
}
