# Dara desktop UI rules (always on)

The Dara desktop app lives in `app/`. The handbook is in `docs/ui/`; read `docs/ui/README.md` and the file for your task before writing code. These rules are the short version; the handbook wins on any conflict.

- Stack is fixed: Tauri 2, Svelte 5 (runes), Vite, TypeScript strict, Tailwind v4. Do not add frameworks or heavy dependencies; justify any new package in the PR.
- Colors: purple primary, white secondary, black tertiary. Use only the semantic tokens from `docs/ui/tokens.css` (`--color-*`). Never write hex/rgb/hsl in components. Never hand-edit `tokens.css`; change `tools/gen_ui_tokens.py` and rerun it.
- All HTTP goes through `app/src/lib/api/client.ts`; types come from `docs/ui/dara-api.d.ts`. Never show a number the API did not return (no computed context, speed or quality). Display "not measured" when the API says so.
- Security: no `{@html}` or `innerHTML` except the sanitized Markdown component; no secrets or message content in logs, URLs or storage; minimal Tauri capabilities; follow `docs/ui/05-security.md`.
- Accessibility: keyboard-only use, visible focus, labels on every control, both themes, reduced motion, 125 % and 150 % scaling.
- Memory: server + UI stay under 300 MB RAM; no timers, polling or animation when nothing is visible.
- Scope: edit only `app/` and `docs/ui/` unless asked. Do not change the C++/CUDA backend; request backend changes instead (list in `docs/ui/07-roadmap-and-tasks.md`).
- Work one milestone at a time (`docs/ui/07-roadmap-and-tasks.md`); run the checks and stop for review when its acceptance criteria are met.
- Do not copy another product's look, wording or branding.
