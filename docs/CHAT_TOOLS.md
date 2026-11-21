# Chat with connectors (tools and agents in chat)

Status: design, nothing built yet. Work happens on `dev`.

## Goal

In the chat page you can turn connectors on for a conversation, ask for something ("what is on my calendar today, and do I have mail about it?"), and Dara's model uses the connectors' tools to do it, several at once where it can, and answers with what it found.

## What the engine gives us (and does not)

- `POST /v1/chat/completions` takes `system`, `user` and `assistant` messages only. There is **no tool-calling support**: no `tools` field, no `tool` role, no constrained decoding.
- Requests run **one at a time** (a queue). Two agents cannot decode at the same moment.

So tool use is built **in the app**, on top of plain text, and "parallel" has a precise meaning here (below).

## Tool protocol (in the prompt, parsed in the app)

1. When connectors are on, the system prompt lists the allowed tools (name, one-line description, parameters) and tells the model to answer with a block like:
   `<tool_call>{"name": "calendar.list_events", "arguments": {...}}</tool_call>`
   one block per call, several blocks allowed in one reply.
2. The app parses blocks out of the finished reply. A block that is not valid JSON, names an unknown tool, or fails the tool's input schema is **not run**: the model gets a short error back and tries again (at most 2 repairs per turn).
3. Tool results go back as a `user` message in a fixed wrapper (`<tool_result name=...>...</tool_result>`), truncated to a size limit, and marked as **data, never instructions**: a mail body that says "ignore your rules" is content.
4. The loop ends when a reply has no tool block, the step limit (default 8) is reached, or the user presses Stop.

Which models can do this well is measured, not assumed: a small test set per model (valid call rate, right tool chosen). Models that fail it do not get the Connectors toggle.

## Parallel: what it means

- **Parallel tool calls (built first).** One reply can ask for several calls; the app runs them concurrently (`join_all`, each through the existing permission and audit path) and sends all results back in one message. Calendar and Gmail lookups overlap, which is where the time goes.
- **Parallel agents (built second).** A "sub-agent" is a separate short conversation with its own task, tool subset and step limit; its final text goes back to the main chat as a tool result. The model decides to start one with a `spawn_agent` call. Because the engine is single-flight, sub-agents **take turns on the model** and overlap only while their tool calls run. That is still useful (agent A waits on Gmail while agent B decodes) and it is honest in the UI: each agent shows "waiting for the model" or "running a tool".
- A real second model instance is out of scope: one model is loaded at a time.

## Safety (same rules as connectors, applied to chat)

- Every call goes through the permission check: **Block** never runs; **Needs approval** opens an approval dialog (connector, tool, its description, the arguments; Deny is focused, Escape denies, Stop closes it; "1 of N" when several wait) and the call row says "Waiting for your approval"; **Allow** runs. A server's own hints are not trusted.
- The model cannot change permissions, add connectors or raise a tool's level.
- Approval cards are per call; "allow for this conversation" is a user choice, never the model's.
- Results are scrubbed of anything that looks like a secret before they reach the model or the log; tool output is shown to the user collapsed under the step it came from.
- Step limit, per-call timeout, result size cap and a cap on calls per turn apply to every agent.

## UI

- Composer: a Connectors button (like Claude's) listing the connected ones, with a switch per connector for this conversation; a tool count and a "needs approval" hint.
- Messages: a collapsible "Used Google Calendar: list events" row per call (arguments, result, time, status), the approval dialog, an agent panel while sub-agents run (each with state and Stop).
- Conversations remember which connectors were on.

## What is stored

- Each assistant message that made calls keeps `stats.tool_calls` (app/src/lib/chat/tools/records.ts): id, name, connector, tool, arguments (each capped at about 2000 characters, `{"_truncated": true, "preview": ...}` above that), status, decision (`auto`, `approved_once`, `approved_always`, `denied_by_user`, `blocked`, `skipped`), start time and duration. The whole stats object stays under the engine's 64 KB: arguments are dropped first, then the last calls. Rows render from it after a restart, with results from the tool-results message.
- Each conversation keeps `params.connectors` (connector ids on in it) and `params.allowed_tools` (`[connectorId, tool]` pairs allowed for the rest of the chat). Only the choice made before a chat exists stays in localStorage (`__new__`).

## Build order

1. Rust: a `call_tools` command that runs a batch of calls concurrently and returns results in order (permission, audit and timeout included); tests with the mock MCP server.
2. TS: tool-protocol module (prompt builder, parser, repair, result wrapper) with unit tests, including hostile inputs.
3. Chat loop in `useChatStore`: multi-turn tool loop, step limit, abort, approval wait, persisted tool rows.
4. UI: Connectors button, call rows, approval cards.
5. Per-model tool test set; gate the toggle.
6. Sub-agents and the agent panel.

## Open questions

- Which loaded models pass the tool test (needs a real run on the dev GPU).
- Whether to store tool rows in the conversation database (needs a schema change in the engine's store, which only accepts `system`, `user` and `assistant` roles today) or keep them app-side.
