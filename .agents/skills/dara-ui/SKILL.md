---
name: dara-ui
description: Build or change the Dara desktop app UI (Tauri + Svelte) in app/. Use for any screen, component, API call, styling, or security question about the Dara frontend.
---

# Dara UI

1. Read `docs/ui/README.md`, then only the files your task needs:
   - what to build and which endpoints a page uses: `docs/ui/01-product-and-pages.md`
   - colors, type, components, accessibility: `docs/ui/02-design-system.md` and `docs/ui/tokens.css`
   - structure, state, streaming, performance budget: `docs/ui/03-architecture-and-stack.md`
   - request/response shapes: `docs/ui/04-backend-api.md`, `docs/ui/dara-api.d.ts`, real responses in `docs/ui/samples/`
   - security: `docs/ui/05-security.md`
   - code style, tests, definition of done: `docs/ui/06-coding-standards.md`
   - what to do next: `docs/ui/07-roadmap-and-tasks.md`
2. Use the tokens (`--color-*`), the components in `app/src/lib/ui/`, and the API client in `app/src/lib/api/`. Do not invent colors, endpoints or numbers.
3. For new data from the server, record or reuse a sample in `docs/ui/samples/` and add it to the mock server so tests run without a GPU.
4. Before finishing: run `npm run check`, `npm run lint`, `npm run test`, `npm run e2e`; walk the definition of done in `docs/ui/06-coding-standards.md`; check both themes and keyboard use.
5. If the UI needs something the backend does not provide, stop and write it up as a backend request instead of working around it.
