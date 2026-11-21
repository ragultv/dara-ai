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
| Connectors page | `app/src/routes/ConnectorsPage.tsx`, `app/src/lib/connectors/` | add, connect, per-tool Ask/Allow/Block, try a tool, call history |
| Workflow engine, scheduler | Rust (step 2 and 4) | |
| LLM steps | the runtime, over its HTTP API | the runtime only ever sees one step |

Data flow in one sentence: **the model runs on this computer; only tool calls and their arguments reach a connector.** A workflow's manifest says which connectors and tools it uses so the user can see what leaves.

## Connectors

- **MCP transports.** stdio (a program on this computer, started without a shell, minimal environment) and Streamable HTTP (https, or this computer; no redirects are followed). The client declares no capabilities, so a server cannot ask Dara to sample, list roots or elicit.
- **Permissions are the user's.** Each tool is Ask, Allow or Block and starts on **Ask**. A server's own `readOnlyHint` is shown ("says read-only") but never grants anything.
- **Pinning.** Each tool has a fingerprint (hash of its name, description, schema and annotations). If a tool changes after the user chose a permission it goes back to Ask and is marked "changed". A tool that disappears keeps the user's choice.
- **Local programs are code.** Adding a stdio connector shows the command and requires an explicit "I trust this program". Variables for it are kept in the credential store; the database keeps only their names.
- **Results are untrusted text.** The UI shows them as plain text, never as HTML or Markdown. LLM steps that read them get no tools.
- **OAuth** (next): HTTP servers that need sign-in use OAuth 2.1 with PKCE, a loopback redirect and the system browser, dynamic client registration and metadata discovery where the server offers them; tokens go in the credential store. A 401 puts the connector in `needs_auth`.
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
2. **OAuth for HTTP connectors** and the credential store.
3. **Workflow format and engine** with tool, transform and output steps only, run against a fake connector so it needs no sign-in. Run history.
4. **LLM steps** with schemas (validate and repair-retry) and token budgeting. In parallel, scope constrained decoding in the runtime.
5. **Scheduler**, catch-up on wake, sidecar health check after resume.
6. **Built-in Google and Microsoft connectors** (decision needed first: whose OAuth client IDs, and Google's verification rules for Gmail scopes), then the **Daily summary** template tested on your own mailbox.
7. **Natural-language authoring** (validated against the schema, dry-run before saving) and the visual editor.
8. **Export and import** with manifests.

## Open questions

- Which of Gmail, Outlook and Calendar have an MCP server we can rely on, and which need a built-in connector?
- OAuth client registration: Google restricts Gmail scopes (verification, and short-lived tokens while an app is in testing); Microsoft allows a public client with PKCE.
- Where the 9B authoring tier is unavailable (small GPUs), is "edit a template" enough?

## Status

Step 1 is on the branch `mcp-connector`. See the pull request for what was built and tested.
