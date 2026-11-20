# Dara desktop app (Tauri + React)

Handbook: `../docs/ui/README.md`. Read it and the file for your task before coding. Rules summary: `../.agents/rules/dara-ui.md`.

- Stack: Tauri 2, React 19, Vite, TypeScript strict, Tailwind v4, Zustand. Source of colors: `../docs/ui/tokens.css` (generated; never hand-edit).
- Backend is finished; call it only through `src/lib/api/`. Types: `../docs/ui/dara-api.d.ts`. Mock data: `../docs/ui/samples/`.
- Commands (once scaffolded in M0): `npm run dev`, `npm run check`, `npm run lint`, `npm run test`, `npm run e2e`, `npm run tauri dev`.
- Keep server + UI under 300 MB RAM; record measurements in `PERF.md`.
- Current milestone and acceptance criteria: `../docs/ui/07-roadmap-and-tasks.md`.
