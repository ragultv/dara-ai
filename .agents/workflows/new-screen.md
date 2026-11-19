---
description: Add a new screen to the Dara app following the handbook
---

1. Read the page's section in `docs/ui/01-product-and-pages.md` and the endpoints it uses in `docs/ui/04-backend-api.md`. If the page is not specified, write its spec there first and ask for confirmation.
2. Create `app/src/lib/features/<name>/` with a container (uses stores) and presentational components (props in, callbacks out). Add the route.
3. Build loading (skeleton), empty, error and offline states first, then the happy path.
4. Use only components from `app/src/lib/ui/` and `--color-*` tokens. Add a missing component to `ui/` with its kit entry, not inside the feature.
5. Add Vitest tests for the logic, a Testing Library test for keyboard use, and a Playwright test against the mock server; run axe in both themes.
6. Run the checks and the definition of done in `docs/ui/06-coding-standards.md`. Add the screenshots (light and dark) to the PR description.
