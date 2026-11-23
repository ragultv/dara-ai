import { DOCS, RELEASES, REPO } from "../data/facts.ts";

export function Footer() {
  return (
    <footer className="footer">
      <div className="wide">
        <div className="footer-card">
          <div className="footer-about">
            <a className="brand" href="/#top" aria-label="Dara, back to the top">
              <img src="/dara-light.png" alt="" />
              <span>Dara</span>
            </a>
            <p>
              Private AI that runs on your own NVIDIA laptop GPU. An early release: the installer is unsigned and every speed was measured on a 4 GB laptop GPU. We would rather show you a smaller true
              number than a bigger guess.
            </p>
          </div>
          <nav className="footer-col" aria-label="Project">
            <h4>Project</h4>
            <a href={REPO}>GitHub</a>
            <a href={RELEASES}>Releases</a>
            <a href={DOCS}>Documentation</a>
          </nav>
          <nav className="footer-col" aria-label="On this page">
            <h4>Explore</h4>
            <a href="/#results">Results</a>
            <a href="/#flash">How Flash works</a>
            <a href="/#context">Long context</a>
            <a href="/#models">Models</a>
            <a href="/#start">Get started</a>
          </nav>
          <nav className="footer-col" aria-label="Legal">
            <h4>Legal</h4>
            <a href="/privacy.html">Privacy</a>
            <a href="/terms.html">Terms of Service</a>
            <a href={`${REPO}/blob/main/LICENSE`}>MIT License</a>
          </nav>
        </div>
        <div className="footer-legal">
          <span>© 2026 Dara contributors. Released under the MIT License.</span>
          <span>Runs on your computer. Nothing leaves it.</span>
        </div>
        <div className="wordmark" aria-hidden="true">
          <img src="/dara-light.png" alt="" />
          <span>Dara</span>
        </div>
      </div>
    </footer>
  );
}
