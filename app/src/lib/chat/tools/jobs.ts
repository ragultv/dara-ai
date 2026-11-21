import { MAX_WAIT_SECONDS, WAIT_TOOL, type ToolResultItem } from "./types.ts";

// Work that is not done when the tool call returns, for ANY connector (nothing here knows a particular server): a tool starts a
// job and says "pending, check again in 15 seconds", or hands back something that needs one more call to become the answer.
// A small model often ends its reply there ("I will check back shortly"). These helpers let the chat loop notice that and send the
// model back to finish, a few times at most.

/** Most times per user message the loop sends the model back to finish unfinished work. */
export const MAX_NUDGES = 3;

const STATUS_VALUE =
  /["']?(?:status|state|job[_ ]?status|progress)["']?\s*[:=]\s*["']?(pending|queued|queueing|in[_ -]?progress|processing|running|started|submitted|generating|waiting|scheduled|created|accepted)\b/i;
const STILL_WORKING =
  /\b(?:still|currently)\s+(?:processing|running|generating|working|in progress|pending|being (?:created|generated|processed))\b/i;
const NOT_READY =
  /\b(?:not\s+(?:yet\s+)?(?:ready|complete[d]?|finished|done)|check(?:ing)?\s+(?:back|again|the\s+status)|wait\s+(?:at\s+least\s+)?\d+\s*(?:s\b|sec)|poll\b|try\s+again\s+(?:later|in)|retry\s+after|come\s+back\s+(?:later|in))/i;
const NEXT_STEP =
  /\b(?:next\s+step|follow[- ]?up\s+(?:call|request)|to\s+(?:complete|finish|get|retrieve|fetch|obtain|continue)\b[^.\n]{0,80}\b(?:call|use|run|invoke)\b)/i;
const SECONDS =
  /\b(?:wait|retry|check|again|after|in|at\s+least)\b[^.\n\d]{0,40}(\d{1,3})\s*(?:s\b|sec|seconds)/i;
const MS = /\bretry[_-]?after[_-]?ms["']?\s*[:=]\s*(\d{3,6})/i;

export interface Unfinished {
  /** how long the tool said to wait, when it said (capped); null when it did not */
  seconds: number | null;
}

/** Does any successful result say the work is not done yet? */
export function unfinishedWork(items: ToolResultItem[]): Unfinished | null {
  let found = false;
  let seconds: number | null = null;
  for (const it of items) {
    if (it.status !== "ok" || it.name === WAIT_TOOL) continue;
    const t = it.text;
    if (!(STATUS_VALUE.test(t) || STILL_WORKING.test(t) || NOT_READY.test(t) || NEXT_STEP.test(t)))
      continue;
    found = true;
    const ms = MS.exec(t);
    const s = SECONDS.exec(t);
    const n = ms ? Math.ceil(Number(ms[1]) / 1000) : s ? Number(s[1]) : null;
    if (n !== null && Number.isFinite(n) && n > 0)
      seconds = Math.max(seconds ?? 0, Math.min(MAX_WAIT_SECONDS, n));
  }
  return found ? { seconds } : null;
}

const PROMISE = [
  /\b(?:i(?:'ll| will| am going to|'m going to)|let me|i can|i shall)\s+(?:now\s+)?(?:check|poll|look|see|follow up|get back|come back|retrieve|fetch|wait|verify)/i,
  /\bcheck(?:ing)?\s+(?:back|again|the status)\b/i,
  /\b(?:once|when)\s+(?:it(?:'s| is)|the\s+\w+\s+is)\s+(?:ready|done|complete|finished)\b/i,
  /\b(?:in a (?:moment|few seconds)|shortly|momentarily)\b/i,
  /\b(?:please\s+)?wait\s+(?:at\s+least\s+)?\d+/i
];

/** Does a reply that made no call promise more work ("I will check back shortly")? */
export function promisesMore(reply: string): boolean {
  return PROMISE.some((p) => p.test(reply));
}

/** The message that sends the model back: shown to no one, it only follows the tool results in the history. */
export function nudgeText(unfinished: Unfinished | null): string {
  const wait = unfinished?.seconds ?? 15;
  return (
    `The work you asked a tool for is not finished, and you cannot check back after your reply ends. ` +
    `Carry on now, with a block and nothing else: write a ${WAIT_TOOL} block for ${wait} seconds, then, after it in the same reply, ` +
    `the tool call that checks or completes it (use any job id or other value the earlier result gave you). ` +
    `Repeat until it is done, then answer with the result, including any link. ` +
    `If it needs something only the user can give, ask for that instead.`
  );
}
