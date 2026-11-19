# Coding standards

## General

- TypeScript `strict: true`, `noUncheckedIndexedAccess: true`, `exactOptionalPropertyTypes: true`. No `any` (use `unknown` and narrow); no `@ts-ignore` (use `@ts-expect-error` with a reason, and rarely).
- Prettier (default config, 100 columns, single quotes off, semicolons on) and ESLint (`@typescript-eslint` strict + `eslint-plugin-svelte` + `jsx-a11y`-style Svelte a11y rules as errors). Rust: `cargo fmt` and `cargo clippy -D warnings`.
- Names: files `kebab-case.ts`; Svelte components `PascalCase.svelte`; stores `name.svelte.ts`; types `PascalCase`; functions and variables `camelCase`; constants `SCREAMING_SNAKE` only for true constants. Booleans read as questions (`isLoading`, `hasImage`).
- Small files: a component over about 250 lines or a function over about 40 gets split. One responsibility per module.
- Comments explain *why* (a constraint, a measured number, a server quirk), not what. No commented-out code. TODOs carry an issue or milestone: `// TODO(M3): ...`.
- No dead code, no unused exports, no console output in committed code (use the logger in `util/log.ts`, which never logs message content or keys).
- Money/units: format bytes, durations and rates only through `util/format.ts` (one place, tested). Numbers shown to users: tok/s with one decimal, context in thousands (`32K`, `128K`) with exact value on hover, memory in MB/GB.

## API and data

- All HTTP goes through `lib/api/client.ts`; types come from `types.ts` (copied from `docs/ui/dara-api.d.ts`). Never import `fetch` in a component.
- Validate server responses at the boundary only where a wrong shape would crash the UI (use small hand-written guards, not a heavy schema library). Tolerate unknown and missing optional fields.
- Errors: throw `ApiError` (`status`, `type`, `message`, `code`, `download`); catch at the store or page level and show an `ErrorState`/toast with the server's message. Never swallow errors silently; never show raw stack traces to users.
- Cancellation: every long request takes an `AbortSignal`. Stores cancel on navigation or when superseded.
- No derived server facts in the UI (what fits, speeds, quality): display what the API returns.

## Svelte

- Svelte 5 runes. `$state` for local state, `$derived` for computed values (not `$effect` that sets state), `$effect` only for side effects with cleanup. Props typed with an `interface Props`.
- Components are presentational where possible: data in through props, events out through callback props. Stores are used by feature containers, not by every leaf.
- Accessibility lint rules are errors, not warnings. Every interactive element is a real `button`/`a`/`input`; icons-only controls have `aria-label`; every form control has a label.
- Never `{@html}` except for the sanitized Markdown output component, which is the single audited place.
- Keyed `{#each}` blocks always (`(item.id)`).

## CSS

- Tailwind utilities with the token-backed theme (`bg-surface`, `text-muted`, `border-border`, `bg-primary`). If a utility does not exist for a token, add it to the theme, do not write the hex.
- No hex, `rgb()`, `hsl()` or `oklch()` literals in `.svelte`/`.css` outside `tokens.css`. A lint rule (stylelint `color-no-hex` + a grep in CI) enforces it.
- Layout with flex/grid and `gap`; spacing from the 4 px scale; no magic pixel numbers for text widths; no `!important`.
- Support both themes in every component (test in both); support `forced-colors`.
- Animations use tokens (`--duration-*`, `--ease`) and respect reduced motion.

## Tests

| Level | Tool | What |
| --- | --- | --- |
| Unit | Vitest | `util/format.ts`, SSE parser, API client error mapping, stores' logic, Markdown sanitizer (with XSS payloads) |
| Component | Vitest + Testing Library | Each ui/ component: keyboard use, ARIA, both themes render |
| End to end | Playwright against the mock server (`app/mocks`, serving `docs/ui/samples`) | First run, download, load progress, chat streaming, stop, options screen, error states |
| Accessibility | axe via Playwright | No serious/critical violations on every page, in both themes |
| Real server | Manual script | One pass against the real `dara.exe` before each milestone is called done |

Every bug fix adds a test that fails without the fix. Mock data comes from `docs/ui/samples`; when the API changes, re-record the samples instead of editing them by hand.

## Git and review

- Branch from `ui/desktop-app`; small PRs (one screen or one concern). Conventional commit subjects: `feat(chat): ...`, `fix(options): ...`, `docs(ui): ...`, `test(...)`, `chore(...)`. Imperative, under 72 characters; the body says why.
- Never commit secrets, `node_modules`, build output, models, or personal paths. Never rewrite pushed history on shared branches.
- The PR description lists what changed, screenshots (light and dark) for UI, and the checklist below.

### Definition of done (every PR)

- [ ] `npm run check` (types), `npm run lint`, `npm run test`, `npm run e2e` pass; `cargo clippy` clean
- [ ] Works with keyboard only; axe clean; both themes; 125 % and 150 % scaling; minimum window size
- [ ] Loading, empty, error and offline states exist and were looked at
- [ ] Uses only tokens; no hex values; no new dependency without a reason
- [ ] Security checklist in 05 satisfied
- [ ] No number shown to the user that the API did not provide
- [ ] Idle CPU near 0 % (the app sitting on each page for a minute); memory measured if the page is new, noted in `app/PERF.md`
- [ ] Docs updated when behavior or API usage changed
