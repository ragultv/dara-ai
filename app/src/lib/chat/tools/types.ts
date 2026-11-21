// Connectors in chat (docs/CHAT_TOOLS.md). The engine has no tool calling, so the app runs the loop on plain text:
// the model writes <tool_call>{"name": ..., "arguments": {...}}</tool_call> blocks, the app runs them (in parallel, through the
// connector permission rules) and sends the results back as a user message wrapped in <tool_results>...</tool_results>.
// Both live in ordinary message content, so conversations persist with no database change and render from content alone.

import type { ToolPermission } from "../../connectors/types.ts";

/** One tool the model may use in this conversation. `name` is what the model writes; it maps back to connector + tool. */
export interface ChatTool {
  /** `<connector slug>__<tool name>`, only [a-z0-9_] , unique in the list */
  name: string;
  connectorId: string;
  connectorName: string;
  /** the tool's own name on the MCP server */
  tool: string;
  title: string | null;
  description: string;
  inputSchema: Record<string, unknown>;
  /** "deny" tools are never offered to the model */
  permission: Exclude<ToolPermission, "deny">;
}

/** A call parsed from a model reply. */
export interface ParsedCall {
  /** stable within the reply: `${assistantMessageId}:${index}` */
  id: string;
  /** the name as the model wrote it */
  name: string;
  arguments: Record<string, unknown>;
}

/** A block the app could not use; sent back to the model so it can fix it. */
export interface ParseProblem {
  index: number;
  /** short, plain reason, e.g. "not valid JSON", "unknown tool foo", "missing required argument 'start'" */
  reason: string;
  raw: string;
}

export interface ParseResult {
  /** the reply's text with every tool_call block removed (what the user sees as prose) */
  text: string;
  calls: ParsedCall[];
  problems: ParseProblem[];
}

export type CallStatus =
  | "waiting_approval"
  | "running"
  | "ok"
  | "error"
  | "denied" // blocked by the user's rule, or refused at the approval card
  | "skipped"; // stopped before it ran (Stop pressed, step limit)

/**
 * Who let a call run, or why it did not:
 * "auto" = allowed by its rule or by an earlier "always allow in this chat"; "approved_once" / "approved_always" = the user's
 * answer to the approval; "denied_by_user" = refused there; "blocked" = the user's rule says deny; "skipped" = stopped.
 */
export type CallDecision =
  "auto" | "approved_once" | "approved_always" | "denied_by_user" | "blocked" | "skipped";

/** Live and finished state of one call, kept in useChatToolsStore; once the step ends it is stored with the message (records.ts). */
export interface CallState {
  id: string;
  conversationId: string;
  name: string;
  connectorId: string | null;
  connectorName: string;
  tool: string;
  arguments: Record<string, unknown>;
  status: CallStatus;
  decision?: CallDecision | undefined;
  /** result text as sent to the model (truncated), or the error */
  output?: string | undefined;
  startedAt?: number | undefined;
  durationMs?: number | undefined;
}

/** One result inside a <tool_results> message. */
export interface ToolResultItem {
  id: string;
  name: string;
  status: "ok" | "error" | "denied" | "skipped";
  /** already truncated to MAX_RESULT_CHARS */
  text: string;
}

export const MAX_STEPS = 8; // model turns per user message
export const MAX_CALLS_PER_TURN = 6;
export const MAX_RESULT_CHARS = 6000;
export const MAX_REPAIRS = 2;
export const CALL_TIMEOUT_MS = 60_000;
