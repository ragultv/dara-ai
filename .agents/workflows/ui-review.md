---
description: Review UI changes against the Dara handbook before a commit or PR
---

Review the current diff against `docs/ui/06-coding-standards.md` (definition of done), `docs/ui/05-security.md` (checklist) and `docs/ui/02-design-system.md`. Check specifically and report each as pass/fail with file and line:

1. Hex/rgb/hsl colors or `opacity` used as a color outside `tokens.css`.
2. `dangerouslySetInnerHTML` / `innerHTML` with anything not produced by the sanitizer; secrets or message content in logs, URLs or storage; new Tauri permissions.
3. `fetch` outside `lib/api/`; numbers shown that the API did not return.
4. Missing loading/empty/error/offline states; missing labels, focus styles, keyboard paths; dark theme not checked.
5. Timers, polling or animation running while the view is hidden.
6. Re-render problems: whole-store selectors, high-frequency state in shared stores, unmemoized list rows, effects used for derived state, missing cleanup.
7. New dependencies and their justification; files over the size guideline.

Then run `npm run check`, `npm run lint`, `npm run test`, `npm run e2e` and report the results. Do not fix anything unless asked.
