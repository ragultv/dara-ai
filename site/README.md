# Dara website

A single-page, editorial site (narrow reading column, wide figures and tables, in the style of a research write-up) in the app's own light theme. It uses `src/tokens.css` (generated from `docs/ui/tokens.css` by `tools/gen_ui_tokens.py`; do not edit by hand), the semantic `--color-*`
tokens only (no hex colors in components), system fonts, borders instead of shadows, and purple only for primary actions and data. The design rules are in `docs/ui/02-design-system.md`.

```powershell
cd site
npm install
npm run dev        # http://localhost:3000
npm run build      # -> site/dist (the home page, privacy.html and terms.html)
npm run check      # type-check (tsconfig.check.json)
npm run data       # regenerate src/data/benchmarks.json from the recorded benchmark runs
```

## Rules for the content

1. **No number on the page is typed in by hand.** Measured numbers come from `src/data/benchmarks.json`, which `scripts/make_data.py` builds from the result files of
   `bench/husky_bench.py`, `bench/context_sweep.py` and `bench/prefill_ab.py` (under `build/bench/`) and from a second machine's results when `RTX-3050-4GB-bench/` is present (it is not tracked in git). Ratios, averages and counts are computed in the components from those rows.
   After a new benchmark run: `npm run data`, then `npm run build`.
2. Everything else (model files and sizes, minimum VRAM, how the app is installed, the API) comes from the repository: `src/serve/catalog.cpp`, `docs/RELEASE.md`, `docs/SERVER.md`. See
   `src/data/facts.ts`.
3. State what was measured on which machine, and what was not. Right now every speed is from a laptop with a 4 GB GPU, and the site says only that; do not claim other GPUs until someone has run
   `docs/RTX_LAPTOP_CHECK.md` on them. Do not hide results that are worse than llama.cpp.
4. Do not claim bit-for-bit identical output for Flash (the default int8 mode can differ from plain decoding at near-tied words), a license (the repository has none yet), a package-manager
   install command, or any speed that no benchmark file records.

## Deploying (Vercel)

Import the repository in Vercel and set **Root Directory** to `site`. `vercel.json` sets the framework (Vite), `npm ci`, `npm run build`, the output folder `dist`, clean URLs (`/privacy`, `/terms`) and security headers. Every push to the branch you choose as the production branch publishes the site; other branches get preview URLs. The site needs no environment variables.
