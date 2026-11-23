# Connectors and workflows

Dara can read your calendar and mail through **connectors** (MCP clients) and run **workflows** (a fixed list of steps, some of which use a local model) on a schedule. The model never decides what to call; steps do.

```
Connectors (MCP and built-in)  ->  Workflow engine  ->  Dara runtime (LLM steps)
 Gmail, Outlook, Calendar...       triggers, steps,       one request at a time,
                                   run history,           interactive requests first
                                   approvals
```

Status: **step 1 (connector core) is built**; see [Status](#status). The rest is the corrected plan below.

## Corrections to the first plan

The architecture holds. These are the places where the plan, checked against how Dara actually works, was wrong or left a gap.

| The plan said | What is wrong | Corrected |
| --- | --- | --- |
| "The engine lives in the desktop app layer" | Ambiguous. The webview is not always running, cannot start programs, and has a locked-down CSP. | The engine, the MCP client, the scheduler and the OAuth listener live in the **Rust layer** (Tauri). The UI is only a client of commands and events. Network calls to connectors come from Rust, so the webview CSP stays `connect-src 'self' 127.0.0.1`. |
| "Stores everything in SQLite" | Dara already has a SQLite database, owned by the C++ server. Two writers on one file, or the runtime knowing about workflows, would couple them. | The desktop layer has **its own SQLite file** (`connectors.db`, later workflow tables). The server database stays the runtime's. Cost: "Delete all data" and export must cover both files. |
| `model: nano` on one step, `model: core` on another | The runtime keeps **one model loaded**. A swap takes seconds, uses the GPU, and interrupts chat. A 4 GB card cannot hold the 9B tier at all. | A run uses **one model**: whatever is loaded, with an optional minimum the workflow requires. Steps do not name tiers. If another model is needed the run waits for an idle moment and is told so. Authoring with the 9B tier is offered only when it is installed and fits. |
| "Output schema enforced by constrained decoding" | The runtime has **no constrained decoding** (no `response_format`, grammar or tool support in `src/serve`). | Two stages: (a) now, validate against the schema, repair-retry a bounded number of times, and record a failed step honestly; (b) later, add schema-constrained sampling to the engine (it touches the sampler, speculative-decoding acceptance and CUDA graphs, so it is its own project). The workflow format does not change between (a) and (b). Keep schemas simple: flat objects, enums, booleans, numbers, short strings, arrays of those. |
| "Interactive requests preempt background runs; the step retries" | The runtime is **single-flight** (a second request gets 503). The only preemption it has is cancel-on-disconnect, and a cancelled step loses the prefix cache. | The engine owns a **priority queue**. Background steps run when chat has been idle for a few seconds; if chat arrives mid-step the step is cancelled (closing the request) and re-queued. LLM steps are small and idempotent so a retry costs only a re-prefill. |
| "If the laptop was asleep at 07:30, it runs on wake" | Only true if the app is running. Also, Windows Modern Standby has dropped the dev GPU before. | Needs start-at-login and the tray (both already settings). On resume the engine **health-checks and restarts the sidecar** before any LLM step. Each workflow sets `catch_up` (`once` or `none`) and `max_late` so a morning summary is not delivered at 6 pm. Not running at all: catch up at the next launch, same rules. |
| Connectors "Gmail, Outlook, Calendar, Canva" as MCP servers | MCP servers for Gmail and Outlook are not guaranteed to exist, and the ones that do differ in tool names and arguments. | A connector is one of three kinds: **MCP over stdio**, **MCP over Streamable HTTP**, or **built in** (Gmail API, Microsoft Graph, Google Calendar, written by us behind the same tool interface). Workflows call **capabilities** (`calendar.list_events`, `mail.search_messages`) with fixed input/output schemas; an adapter per connector maps them. MCP stays the extension mechanism. Which vendors host MCP servers today must be checked before step 5. |
| `key: ical_uid` for merging calendars | Recurring events share a UID; Google and Outlook expose different fields. | Merge key is `(uid, start)` with a fallback of normalized title plus start time. |
| YAML or JSON | YAML 1.1 reads `07:30` as a number and `no` as false; aliases allow billion-laughs files. | **JSON is canonical** and is validated by a JSON Schema. YAML is accepted only on import (YAML 1.2 core, aliases refused) and converted. |
| Placeholder tool names | A name that is wrong at 07:30 is a failed run. | Tool references are `connector.tool` and **validated against the discovered tool list when the workflow is saved**, and against the pinned tool fingerprint when it runs. |
| "Engine counts tokens per step" | Needs a tokenizer. | The runtime already has `POST /dara/tokenize`; use it. Truncation is never silent: a step that had to cut its input records that, and the output says so. |
| Safety list | Missing: what an MCP server itself can do. | Added below: stdio servers are programs, tool descriptions and results are untrusted, a server's `readOnlyHint` is never trusted, tool lists are pinned. |

## What lives where

| Piece | Where | Notes |
| --- | --- | --- |
| MCP client (stdio, Streamable HTTP), connector manager, permissions, audit | Rust crate `app/src-tauri/crates/dara-mcp` | headless (no Tauri dependency), so the engine can use it without a window |
| Tauri commands and events | `app/src-tauri/src/connectors.rs` | the only way the UI reaches connectors |
| Secrets | Windows Credential Manager (service `Dara`) | never in SQLite, never in exports, deleted on disconnect |
| Connector database | `%LOCALAPPDATA%\Dara\connectors.db` | connectors, per-tool permissions, audit; later workflows and runs |
| Connectors (Settings > Connectors) | `app/src/lib/features/connectors/`, `app/src/lib/connectors/` | an apps view: cards from a catalog plus your own, a page per connector (Overview, Permissions), sign-in, per-tool Allow / Needs approval / Block, try a tool |
| Workflow engine, scheduler | Rust (step 2 and 4) | |
| LLM steps | the runtime, over its HTTP API | the runtime only ever sees one step |

Data flow in one sentence: **the model runs on this computer; only tool calls and their arguments reach a connector.** A workflow's manifest says which connectors and tools it uses so the user can see what leaves.

## Connectors

- **MCP transports.** stdio (a program on this computer, started without a shell, minimal environment) and Streamable HTTP (https, or this computer; no redirects are followed). The client declares no capabilities, so a server cannot ask Dara to sample, list roots or elicit.
- **Permissions are the user's.** Each tool is Ask, Allow or Block and starts on **Ask**. A server's own `readOnlyHint` is shown ("says read-only") but never grants anything.
- **Pinning.** Each tool has a fingerprint (hash of its name, description, schema and annotations). If a tool changes after the user chose a permission it goes back to Ask and is marked "changed". A tool that disappears keeps the user's choice.
- **Local programs are code.** Adding a stdio connector shows the command and requires an explicit "I trust this program". Variables for it are kept in the credential store; the database keeps only their names.
- **Results are untrusted text.** The UI shows them as plain text, never as HTML or Markdown. LLM steps that read them get no tools.
- **API keys** (built, module `apikey` in the crate): for a server that has no sign-in, only a key. The user pastes it (when adding a custom server address, on the connector's page later, or in the dialog a catalog entry with `"registers": "api-key"` opens). The key lives only in the credential store (`connector/<id>/api_key`), never in the database, the UI payloads or the audit log (it is in the scrub list, so an error or an argument that contains it is redacted). The database keeps only the NAME of the header it is sent in (`api_key_header`, migration 3): empty means `Authorization: Bearer <key>`, a name such as `X-API-Key` sends the key as it is. Connection headers (`Host`, `Content-Type`, `Mcp-Session-Id`, ...) are refused. The key is read from the store on every request, and changing or removing it disconnects the connector. A key beats a sign-in; a 401 with a key is shown as "the server did not accept the API key" (an error), not as a sign-in prompt.
- **OAuth** (built, module `oauth` in the crate): HTTP servers that need sign-in use the MCP authorization flow. Discovery reads the 401's `WWW-Authenticate`, then the protected-resource metadata (RFC 9728), then the authorization-server metadata (RFC 8414, or OIDC discovery). The client is a user-supplied client ID, else Dynamic Client Registration (the registered ID and its loopback port are remembered). PKCE S256, a random `state`, the `resource` parameter (RFC 8707) and the `iss` check (RFC 9207). The sign-in page opens in the system browser and a loopback listener (127.0.0.1, random port, one request, 5-minute timeout, cancellable) receives the answer. Tokens live only in the credential store, in chunks with the header written last, so a half-written token never reads as valid. One refresh at a time; a rotated refresh token is saved before the new access token is used; `invalid_grant` goes to "Needs sign-in" without a retry loop; a 403 `insufficient_scope` shows "Approve more access" and never re-authorizes silently. Every metadata, token and registration address must be https, is not redirected and is refused if it points at a private, loopback (unless the server itself is local), link-local or unique-local address. Signing out revokes at the server when it offers revocation, and deletes the tokens either way.
- **Audit.** Every call (tool, arguments cut at 8 KB, outcome, duration) is kept, at most 5000 rows, and can be purged.

## Workflow format

A workflow is **JSON** with a `version`, validated by a JSON Schema, stored locally and versioned. A step has an `id` and exactly one type:

| Step type | Does |
| --- | --- |
| trigger | schedule with a time zone, manual run, hotkey, or "N minutes before each meeting"; `catch_up` and `max_late` |
| tool | calls one capability or one `connector.tool` with arguments from earlier steps |
| transform | deterministic, from a **closed library** (filter, rank, sort, merge, dedupe, format); no user code |
| llm | prompt template, input references, required output schema; the model is the run's model |
| for_each | runs steps per item; map-then-combine for large inputs |
| branch | conditions on earlier outputs |
| approval | pauses and shows exactly what will happen |
| output | panel card, notification, file or draft |

References between steps use a small expression syntax (`steps.events.output[0].title`) that cannot run code. Schemas are named and versioned (`digest_v1`) and live in the app, so the digest is drawn from typed fields.

## The connector catalog

Settings > Connectors shows suggested connectors as cards. Their texts are in `app/src/lib/connectors/catalog.json` and their logos in `app/src/assets/connectors/`, so a connector is added or edited without touching code (`catalog.ts` only loads them); a connector you add by hand shows up next to them. Every address in the catalog was probed against the live server on 2025-11-21 (an unauthenticated MCP request, then the OAuth metadata it points to):

| Connector | Address | Sign-in | Registers Dara by itself? |
| --- | --- | --- | --- |
| Google Calendar | `https://calendarmcp.googleapis.com/mcp/v1` | OAuth (Google) | No: your own client ID (Desktop app type) |
| Gmail | `https://gmailmcp.googleapis.com/mcp/v1` | OAuth (Google) | No: your own client ID |
| Canva | `https://mcp.canva.com/mcp` | OAuth | Yes (dynamic registration) |
| Firecrawl | `https://mcp.firecrawl.dev/v2/mcp-search` | OAuth (Firecrawl account) | Yes (dynamic registration, public client) |
| Todoist | `https://ai.todoist.net/mcp` | OAuth | Yes (dynamic registration); discovery checked with `probe_oauth`, sign-in not yet tried |
| Tavily | `https://mcp.tavily.com/mcp` | OAuth | Yes (dynamic registration); discovery checked, sign-in not yet tried |
| Airtable | `https://mcp.airtable.com/mcp` | OAuth | Yes (dynamic registration); discovery checked, sign-in not yet tried |
| Context7 | `https://mcp.context7.com/mcp` | none | n/a: open server, 2 tools, a call answered with no sign-in |
| DeepWiki | `https://mcp.deepwiki.com/mcp` | none | n/a: open server, 3 tools, a call answered with no sign-in |
| Hugging Face | `https://huggingface.co/mcp` | none (public data) | n/a: open server, 4 tools, a call answered with no sign-in |
| Outlook mail and calendar | (built in, not available yet) | | |

Facts worth knowing:

- Google's servers are **stateless**: `initialize` and `tools/list` answer without a token, and only `tools/call` needs one (a 401 whose `WWW-Authenticate` points at the protected-resource metadata, with the tool name in the path). So a Google connector can say "Connected" and still be "not signed in"; the page shows Sign in whenever there are no tokens. Google offers no dynamic registration, so Sign in stays disabled until a client ID is saved.
- Each Google connector asks for an **access level**: Read only (the default; `gmail.readonly`, `calendar.readonly` and `calendar.calendarlist.readonly`) or Read and write / organize (`calendar`; `gmail.modify` and `gmail.compose`, which cannot send). Changing it takes a new sign-in.
- Gmail's tools can search threads and read messages and drafts (`search_threads`, `get_thread`, `get_message`); Calendar's can list events in a date range (`list_events` with start and end time). That fits the daily summary.
- Notion (`mcp.notion.com/mcp`) and Linear (`mcp.linear.app/mcp`) both answer and register Dara themselves, but were taken out of the catalog on request; a connector can still be added by hand with its address.
- Seen but **not** in the catalog yet, because sign-in was not verified for them: Asana (`mcp.asana.com/v2/mcp`, no registration), Slack (`mcp.slack.com/mcp`, no registration), Stripe (`mcp.stripe.com`), Atlassian (`mcp.atlassian.com/v1/mcp`, no resource metadata), Sentry, Supabase, HubSpot. GitHub's server (`api.githubcopilot.com/mcp/`) advertises no PKCE (S256), so it is not usable. A Shopify address I tried answered 404.
- Figma (`mcp.figma.com/mcp`) was tried and taken out: its remote server signs in only clients in Figma's MCP catalog (new ones join a waitlist; dynamic registration answers 403), and a personal access token in `X-Figma-Token` is rejected. Its desktop server (`http://127.0.0.1:3845/mcp`, Dev Mode, needs the desktop app and a Dev or Full seat) has no sign-in and would need a "no sign-in" catalog mode.
- An **open server** has `"registers": "none"` in `catalog.json`: no sign-in and no key, so its page shows "No sign-in needed" and it connects as soon as it is added. It can still refuse a call later (Google's servers list tools freely and want a token to call them); the three above were each called once without a token.
- `cargo run -p dara-mcp --example probe_connect -- [--json out.json] <urls...>` connects through Dara's own connector manager and lists the tools (a live check; `DARA_PROBE_API_KEY` and `--key-header` try a key). `probe_oauth` checks sign-in discovery only. Neither registers a client or stores anything.
- **What the tools cost.** Every request carries all enabled tools in the system prompt (name, description cut to 300 characters, every argument). Measured with `tests/unit/tool-prompt-size.test.ts` on real tool lists (`DARA_TOOL_JSON=<file from probe_connect>`; tokens are the app's own estimate of 3.5 characters each): Context7 415, DeepWiki 238, Hugging Face 853, Firecrawl (3 tools) 1,612, Calendar (9 tools) 2,964, Gmail (23 tools) 4,261; all six together are 44 tools, about 10,500 tokens, which is 32% of a 32K context and 16% of a 64K one. A one-line-per-tool list would be about 1,800 tokens. Todoist, Tavily, Airtable and Canva could not be measured (their tool lists need a sign-in).
- **The compact list** (`shouldUseCompactList` in `chat/tools/protocol.ts`): one line per tool, `name(required arguments, ...): first sentence`, plus Dara's own `dara__describe_tool` so the model can look up one tool's arguments before using it. The same 44 tools take 2,168 tokens instead of 10,525. It is a **last resort**, used only when the full list would take a quarter of the context (or 6,000 tokens when the context is unknown), because it made small models worse. Tried with the real engine and the app's own parser and repair loop, three questions (mail, calendar, docs), three tries each: Dara Nano made a tool call in 9 of 9 tries with the full list and in 3 of 9 with the compact one (the rest apologised or asked the user for more detail); neither model used the lookup step, both guessed from the one-liners, and Core 4B picked less suitable tools (`list_calendars` for "what is on my calendar tomorrow", `query_docs` without resolving the library first). "Made a call" means the call parsed; some full-list calls had empty arguments. Dara Nano runs 131,072 tokens of context on the 4 GB laptop card and Core 4B 37,888, so a list of this size takes about 8% and 23%: the full list is kept.
- Seen in that test and not looked into: Dara Core 4B wrote its calls as bare JSON with no `<tool_call>` tags in a plain chat request (the app sends more, such as the reasoning effort), so the parser read them as no call. Worth checking in the app with Core 4B and a connector.
- A wrong call comes back with the tool's arguments (`toolSignature`), once per tool, so the model's next try can be right.
- Firecrawl has two servers: `/v2/mcp` takes only an API key (`Authorization: Bearer <key>`; without one a tool call says "Anonymous keyless access is unavailable"), `/v2/mcp-search` signs in with OAuth and offers search, scrape, research and agent tools. The catalog uses the sign-in one. A server that has only the API-key kind is added with `"registers": "api-key"` in `catalog.json`.
- Logos: Gmail, Google Calendar and Canva are the single-colour Simple Icons marks in each brand's colour, Firecrawl is the favicon from its own site, and Outlook is a drawn placeholder. Replace a file in `app/src/assets/connectors/` (same name) to change one.
- The cards show no "Verified" mark: Dara does not know who runs a server, only its address ("Can send data to ...").

### Google: one client for the app, not one per user

Google gives an **app** a client ID, not each user. The developer creates one OAuth client (type **Desktop app**) in a Google Cloud project, and every user signs in to it with their own Google account. Dara has no server, so each user's tokens stay on their own computer; the project only identifies the app. A desktop app cannot keep a secret, and Google's rules for installed apps say the client secret is not confidential there (PKCE protects the sign-in), so shipping it in the build is acceptable.

- **In the build, not in the repository.** Set `VITE_DARA_GOOGLE_CLIENT_ID` and `VITE_DARA_GOOGLE_CLIENT_SECRET` (for example in `app/.env.local`, which is git-ignored) when building. With them set, Google connectors are added with that client, the page shows "Dara's Google sign-in" and users see no form; "Use my own" still lets a user bring a different client. Without them, the page asks for a client ID, as before.
- **What stands between this and every user** (from my knowledge of Google's rules; check them in the Cloud console, they change):
  - While the consent screen is in *testing* mode only the test users you list (up to 100) can sign in, and refresh tokens last about 7 days.
  - Before verification, other users get an "unverified app" warning and a lifetime cap of about 100 users.
  - Calendar scopes (`calendar`, `calendar.readonly`) are *sensitive*: verification needs a published privacy policy, a homepage on a domain you control, and a justification for each scope.
  - Gmail scopes (`gmail.readonly`, `gmail.modify`, `gmail.compose`) are *restricted*: verification is stricter and Google may require an independent security assessment; whether an app that keeps all data on the user's device is exempt from the assessment is something to confirm with Google. Start early.
  - The Google API Services "limited use" rules apply: mail may be used only to give the user the feature they asked for. Dara keeps it on the device and does not send it to Dara's own servers.
- API quotas are counted per Cloud project, shared by every user of the app; the per-user limits are separate.

## Safety rules for user-built workflows

- **Manifest.** Each workflow declares the connectors and tools it needs; the user approves it once, and again if it changes.
- **Unattended runs are read-only.** A write needs an approval step, or an explicit per-workflow "auto" setting with limits (drafts only).
- **Taint.** Values from tools are untrusted, and so is anything derived from them, including model output. LLM steps that read them get no tools. A value derived from untrusted data cannot become an argument of a write call without confirmation.
- **Imported workflows** are untrusted: show the manifest and the steps before enabling.
- **History is sensitive.** Run history keeps step inputs, which means mail and calendar text. A retention setting (default 30 days) and a purge button apply, and history is left out of exports unless asked. Cached sample data for dry runs is the same.
- **Audit** of tool calls, approvals and outputs per run.

## Engine behavior

- **Resumable runs.** State is saved after each step; a crash or an interrupting chat message restarts at the last step.
- **Priority queue** with one runtime request in flight; interactive chat first (see the correction above).
- **Reliability.** Timeouts, retries with backoff, pagination, per-connector rate limits.
- **Dry run** against cached sample data, with step-by-step results and timings.
- **Token budget** per step against the loaded model's context, counted with the runtime's tokenizer.

## Build order (revised)

1. **Connector core**: MCP client (stdio, Streamable HTTP), manager, permissions, audit, Connectors page. Tested against mock servers. *(done)*
2. **OAuth for HTTP connectors** and the credential store. *(done)*
3. **Workflow format and engine** with tool, transform and output steps only, run against a fake connector so it needs no sign-in. Run history.
4. **LLM steps** with schemas (validate and repair-retry) and token budgeting. In parallel, scope constrained decoding in the runtime.
5. **Scheduler**, catch-up on wake, sidecar health check after resume.
6. **Built-in Google and Microsoft connectors** (decision needed first: whose OAuth client IDs, and Google's verification rules for Gmail scopes), then the **Daily summary** template tested on your own mailbox.
7. **Natural-language authoring** (validated against the schema, dry-run before saving) and the visual editor.
8. **Export and import** with manifests.

## Open questions, with what is known

None of this has been verified against a live endpoint; it comes from search excerpts and must be checked before step 6.

- **Which connectors have an MCP server?**
  - **Google:** documents remote MCP servers for Workspace (Gmail, Drive, Calendar and others). You enable the APIs in your own Cloud project and sign in with OAuth 2.0. Check on the real endpoint that the tools return what the daily summary needs: events for a date range, and mail search that returns message bodies.
  - **Canva:** `mcp.canva.com/mcp`, Streamable HTTP, OAuth.
  - **Outlook mail and calendar:** nothing found that confirms a Microsoft-hosted server, so plan a built-in Microsoft Graph connector.
- **Whose OAuth client IDs?** Dara's own, because Dara is the application: a Google Cloud project with the Gmail and Calendar APIs enabled, and a Microsoft Entra app registration with read-only Graph permissions (a public client with PKCE).
  - **Beta:** a Google app in testing mode is limited to listed test users, and refresh tokens expire after about 7 days there.
  - **Public release:** Gmail's restricted scopes need Google's verification, which can take a security assessment. Start it early.
- Where the 9B authoring tier is unavailable (small GPUs), is "edit a template" enough?

## Status

**Step 1, the connector core, is built** on the branch `mcp-connector`.

- **`dara-mcp` crate** (`app/src-tauri/crates/dara-mcp`): MCP client (spec 2025-06-18, negotiating down to 2025-03-26 and 2024-11-05) over stdio and Streamable HTTP; connector manager; per-tool permissions with fingerprint pinning; audit; secrets in the Windows Credential Manager. 40 offline tests against mock servers, including a contract test that records the payloads in `docs/ui/samples/connectors.json`.
- **Tauri commands** (`app/src-tauri/src/connectors.rs`): `connectors_list`, `connector_add`, `connector_update`, `connector_remove`, `connector_connect`, `connector_disconnect`, `connector_tools`, `connector_set_permission`, `connector_call`, `connector_audit`, `connector_stderr`. Payloads are camelCase; errors are plain strings.
- **Connectors page** (`app/src/routes/ConnectorsPage.tsx`): add, connect, per-tool Ask/Allow/Block, try a tool, call history. In a browser tab it runs on a demo backend whose output is tested against the recorded sample, so it cannot drift from the real one.
- **OAuth (step 2):** see the Connectors section above. 79 offline tests against a mock authorization server and OAuth-protected MCP endpoint, including the negative cases (wrong state, wrong or missing PKCE verifier, mismatched `iss`, expired code, redirect to a non-loopback address, port collision, a browser that never returns, a rotating refresh, a revoked refresh token, `insufficient_scope`, private or non-https metadata addresses). The Connectors page has Sign in, Cancel sign-in, Sign out, "Approve more access" and sign-in settings (client ID, secret, scope).
- **Not done:** everything from step 3 on, and no real server has been tried (Google's remote server needs your own OAuth client ID; Canva's should register Dara by itself).

Things found while building that the first plan did not say:

- On Windows `npx` is `npx.cmd`; a program started without a shell is not found without the extension, so the command is looked up on PATH with `.exe`, `.cmd`, `.bat` and `.com` (still no shell).
- A child process must not inherit the app's environment (API keys, tokens). It gets PATH, SystemRoot, TEMP, TMP, USERPROFILE, APPDATA, LOCALAPPDATA and HOME, plus the variables the user gave it.
- Streamable HTTP: a POST may be answered with JSON or with an SSE stream, with notifications and requests mixed in; a 404 on a known session means start a new session once. The older HTTP+SSE transport is deprecated and not supported.
- A Windows credential holds about 2.5 KB, so a variable value over 2000 characters is refused.
- If the tool list changed and a tool's definition changed with it, a call that was approved is **not** made: what the user approved is no longer what is there. The reset itself is in the store (`Store::sync_tools`), so it holds for every connector kind; a test covers an HTTP connector, and the fingerprint includes the tool's title as well as its name, description, schema and annotations.

Run the tests: `cd app/src-tauri && cargo test -p dara-mcp` (use `CARGO_TARGET_DIR` outside the repo while `tauri dev` is running) and `cd app && npm run test`.

More found while building OAuth that the first plan did not say:

- The 401 can carry a `scope`; it is sent on the authorization request.
- A random loopback port is a problem for registration, because many servers register an exact redirect address; the registered port is stored and reused when free.
- Microsoft's JWT access tokens can be larger than a Windows credential, so chunking is required.
- A server with no resource metadata is treated as its own authorization server (the older 2025-03-26 behaviour).
- Client ID Metadata Documents are detected, but Dara cannot host one for a desktop app, so it falls through to Dynamic Client Registration.
- If a server's authorization page itself refuses a redirect address, nothing tells Dara; that case ends in the 5-minute timeout.
