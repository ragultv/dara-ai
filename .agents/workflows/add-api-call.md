---
description: Add a call to a Dara server endpoint in the UI
---

1. Find the endpoint in `docs/ui/04-backend-api.md` and `docs/SERVER.md`. If it is not there, stop: the UI does not call undocumented endpoints; write a backend request instead.
2. Make sure its types exist in `app/src/lib/api/types.ts` (source: `docs/ui/dara-api.d.ts`; update both together). Optional fields stay optional.
3. Add a typed function in `app/src/lib/api/` that uses `client.ts` (auth, timeout, `AbortSignal`, `ApiError`). Never call `fetch` from a component.
4. Add a sample response to `docs/ui/samples/` (record it from a real server; redact usernames/paths) and serve it from `app/mocks`.
5. Test: success, each documented error status, cancellation, and a response with extra unknown fields.
6. Use it from a store, with polling only while visible and cleanup on teardown.
