import { useConnectorsStore } from "../../stores/useConnectorsStore.ts";
import { allowKey, useChatToolsStore } from "../../stores/useChatToolsStore.ts";
import type { ToolCallResult } from "../../connectors/types.ts";
import { waitTool, buildToolList, resolveTool, resultToText, toolSignature } from "./protocol.ts";
import {
  CALL_TIMEOUT_MS,
  MAX_RESULT_CHARS,
  type CallDecision,
  type CallState,
  type CallStatus,
  type ChatTool,
  type ParsedCall,
  type ToolResultItem,
  WAIT_TOOL,
  DESCRIBE_TOOL,
  MAX_WAIT_SECONDS
} from "./types.ts";

// Runs the tool calls the model asked for (docs/CHAT_TOOLS.md). Every call goes through the connector permission path:
// "ask" tools wait for the user's approval card, "deny" tools are refused by the desktop layer. The model cannot change either.

/**
 * Makes the conversation's connectors ready and lists their tools for the model.
 * `problems` are short lines for the user, e.g. "Gmail is not signed in".
 */
export async function loadChatTools(
  enabledIds: string[]
): Promise<{ tools: ChatTool[]; problems: string[] }> {
  const problems: string[] = [];
  if (enabledIds.length === 0) return { tools: [], problems };
  const store = useConnectorsStore.getState;
  if (store().entries.length === 0 || enabledIds.some((id) => !findEntry(id))) {
    await store().load();
  }

  await Promise.all(
    enabledIds.map(async (id) => {
      let entry = findEntry(id);
      if (!entry) {
        problems.push("A connector chosen for this chat was removed");
        return;
      }
      const name = entry.config.name;
      if (!entry.config.enabled) {
        problems.push(`${name} is turned off`);
        return;
      }
      if (entry.status.state === "disconnected" || entry.status.state === "error") {
        await store().connect(id);
        entry = findEntry(id);
      } else if (entry.status.state === "connecting") {
        // someone else is connecting it; wait briefly for that to settle
        entry = await waitSettled(id);
      }
      const state = entry?.status.state;
      if (state === "needs_auth") {
        problems.push(`${name} is not signed in`);
        return;
      }
      if (state !== "ready") {
        const why = entry?.status.error;
        problems.push(why ? `${name} could not connect: ${why}` : `${name} could not connect`);
        return;
      }
      if (!store().tools[id]) await store().loadTools(id);
      if (!store().tools[id]) problems.push(`Could not read the tools of ${name}`);
    })
  );

  const s = store();
  const tools = buildToolList(s.entries, s.tools, enabledIds);
  return { tools: tools.length > 0 ? [...tools, waitTool()] : tools, problems };
}

const findEntry = (id: string) =>
  useConnectorsStore.getState().entries.find((e) => e.config.id === id);

async function waitSettled(id: string) {
  for (let i = 0; i < 100; i++) {
    const e = findEntry(id);
    if (!e || e.status.state !== "connecting") return e;
    await new Promise((r) => setTimeout(r, 100));
  }
  return findEntry(id);
}

class Stopped extends Error {}
class TimedOut extends Error {}

/** Resolves with `p`, or rejects when the signal aborts or the time runs out. */
function guard<T>(p: Promise<T>, signal: AbortSignal, timeoutMs: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    if (signal.aborted) {
      reject(new Stopped());
      return;
    }
    const timer = setTimeout(() => {
      cleanup();
      reject(new TimedOut());
    }, timeoutMs);
    const onAbort = () => {
      cleanup();
      reject(new Stopped());
    };
    const cleanup = () => {
      clearTimeout(timer);
      signal.removeEventListener("abort", onAbort);
    };
    signal.addEventListener("abort", onAbort, { once: true });
    p.then(
      (v) => {
        cleanup();
        resolve(v);
      },
      (e: unknown) => {
        cleanup();
        reject(e instanceof Error ? e : new Error(String(e)));
      }
    );
  });
}

const errorText = (e: unknown) =>
  e instanceof Error && e.message ? e.message : "The tool call failed";

const clip = (s: string) =>
  s.length > MAX_RESULT_CHARS ? `${s.slice(0, MAX_RESULT_CHARS)}\n(truncated)` : s;

/** Keeps the conversation's live line in step with its calls. */
function refreshStatus(conversationId: string, ids: string[]) {
  const st = useChatToolsStore.getState();
  let running = 0;
  let waitingN = 0;
  for (const id of ids) {
    const s = st.calls[id]?.status;
    if (s === "running") running++;
    else if (s === "waiting_approval") waitingN++;
  }
  const line =
    waitingN > 0
      ? "Waiting for your approval"
      : running > 0
        ? running === 1
          ? "Running 1 tool"
          : `Running ${running} tools`
        : undefined;
  st.setStatus(conversationId, line);
}

/**
 * Runs every call at once (results keep the calls' order). Never rejects: each call ends as ok, error, denied or skipped.
 */
export async function runCalls(
  conversationId: string,
  calls: ParsedCall[],
  tools: ChatTool[],
  signal: AbortSignal
): Promise<ToolResultItem[]> {
  const chat = useChatToolsStore.getState;
  const ids = calls.map((c) => c.id);

  const runOne = async (call: ParsedCall): Promise<ToolResultItem> => {
    const tool = tools.find((t) => t.name === call.name);
    const base: CallState = {
      id: call.id,
      conversationId,
      name: call.name,
      connectorId: tool?.connectorId ?? null,
      connectorName: tool?.connectorName ?? "",
      tool: tool?.tool ?? call.name,
      arguments: call.arguments,
      status: "running",
      startedAt: Date.now()
    };

    const end = (
      status: Exclude<CallStatus, "running" | "waiting_approval">,
      text: string,
      decision?: CallDecision
    ) => {
      const output = clip(text);
      const prev = chat().calls[call.id];
      const startedAt = prev?.startedAt ?? base.startedAt ?? Date.now();
      // a decision already made (e.g. "approved_once") stays; `decision` fills in or overrides where it is final
      const d = decision ?? prev?.decision;
      chat().patchCall(call.id, {
        status,
        output,
        durationMs: Date.now() - startedAt,
        ...(d ? { decision: d } : {})
      });
      refreshStatus(conversationId, ids);
      return { id: call.id, name: call.name, status, text: output } satisfies ToolResultItem;
    };

    if (!tool) {
      chat().setCall({ ...base, status: "error" });
      return end("error", `Unknown tool ${call.name}. Use only the tools listed.`);
    }
    if (signal.aborted) {
      chat().setCall({ ...base, status: "skipped" });
      return end("skipped", "Stopped before it ran.", "skipped");
    }

    chat().setCall(base);
    refreshStatus(conversationId, ids);

    // Dara's own lookup: answers from the tool list in hand, no connector, no approval
    if (tool.name === DESCRIBE_TOOL) {
      const asked = String(call.arguments["name"] ?? "").trim();
      const listed = tools.filter((t) => t.name !== DESCRIBE_TOOL);
      const target = asked ? resolveTool(asked, listed) : undefined;
      if (!target)
        return end(
          "error",
          `There is no tool ${asked ? `named ${asked}` : "with that name"}. Valid tool names: ${listed.map((t) => t.name).join(", ")}`
        );
      return end("ok", toolSignature(target), "auto");
    }

    // Dara's own wait: no connector, no approval; Stop ends it at once
    if (tool.name === WAIT_TOOL) {
      const raw = Number(call.arguments["seconds"]);
      const secs = Math.min(
        MAX_WAIT_SECONDS,
        Math.max(1, Number.isFinite(raw) ? Math.round(raw) : 10)
      );
      try {
        await guard(
          new Promise<void>((r) => setTimeout(r, secs * 1000)),
          signal,
          (secs + 5) * 1000
        );
      } catch {
        return end("skipped", "Stopped while waiting.", "skipped");
      }
      return end("ok", `Waited ${secs} seconds.`, "auto");
    }

    const invoke = async (approved: boolean): Promise<ToolCallResult> => {
      const r = await guard(
        useConnectorsStore.getState().call(tool.connectorId, tool.tool, call.arguments, approved),
        signal,
        CALL_TIMEOUT_MS
      );
      if ("error" in r) throw new Error(r.error);
      return r;
    };

    try {
      const always = (chat().alwaysAllowed[conversationId] ?? []).includes(
        allowKey(tool.connectorId, tool.tool)
      );
      let result = await invoke(always);
      if (result.outcome !== "needs_approval") chat().patchCall(call.id, { decision: "auto" });

      if (result.outcome === "needs_approval") {
        chat().patchCall(call.id, { status: "waiting_approval" });
        refreshStatus(conversationId, ids);
        const allow = await chat().requestApproval(call.id, signal);
        if (signal.aborted) return end("skipped", "Stopped before it ran.", "skipped");
        if (!allow) return end("denied", "The user did not allow this call.", "denied_by_user");
        // the clock starts again once the user said yes
        chat().patchCall(call.id, { status: "running", startedAt: Date.now() });
        refreshStatus(conversationId, ids);
        result = await invoke(true);
      }

      switch (result.outcome) {
        case "ok":
          return end(result.isError ? "error" : "ok", resultToText(result));
        case "denied":
          return end("denied", "This tool is blocked by the user's settings.", "blocked");
        case "needs_approval":
          return end("denied", "The call was not approved.", "denied_by_user");
        default:
          return end("error", resultToText(result) || "The tool call failed.");
      }
    } catch (e) {
      if (e instanceof Stopped)
        return end(
          "skipped",
          "Stopped before it finished.",
          chat().calls[call.id]?.decision ?? "skipped"
        );
      if (e instanceof TimedOut)
        return end(
          "error",
          `The tool did not answer within ${Math.round(CALL_TIMEOUT_MS / 1000)} seconds.`
        );
      return end("error", errorText(e));
    }
  };

  try {
    // Calls run at the same time, except around a wait: the calls after a wait block run once it is over (check a job's status).
    const out: ToolResultItem[] = [];
    let batch: ParsedCall[] = [];
    for (const c of calls) {
      if (c.name === WAIT_TOOL) {
        out.push(...(await Promise.all(batch.map(runOne))));
        batch = [];
        out.push(await runOne(c));
      } else batch.push(c);
    }
    out.push(...(await Promise.all(batch.map(runOne))));
    return out;
  } finally {
    refreshStatus(conversationId, ids);
  }
}
