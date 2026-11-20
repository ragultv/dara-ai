import { apiClient } from "../api/client.ts";
import type { ChatMessage } from "../api/types.ts";
import type { ReasoningEffort } from "../chat/effort.ts";

export interface CompletionRun {
  messages: ChatMessage[];
  temperature: number;
  maxTokens: number;
  effort?: ReasoningEffort;
  signal?: AbortSignal;
  /** the answer so far (raw), as it streams */
  onContent?: (raw: string) => void;
  /** the thinking so far; `done` once the answer has started */
  onReasoning?: (text: string, done: boolean, ms: number) => void;
}

export interface CompletionResult {
  content: string;
  reasoning: string;
  thinkingMs: number;
  /** the caller stopped it; content and reasoning hold what had arrived */
  aborted: boolean;
}

/** One streamed request to the local engine. Throws on a network or engine error; a stop by the user is a normal result. */
export async function complete(run: CompletionRun): Promise<CompletionResult> {
  let content = "";
  let reasoning = "";
  let thinkStart = 0;
  let thinkingMs = 0;
  let thinkDone = false;
  try {
    const stream = await apiClient.streamChat(
      {
        messages: run.messages,
        temperature: run.temperature,
        max_tokens: run.maxTokens,
        ...(run.effort && run.effort !== "none" ? { reasoning_effort: run.effort } : {})
      },
      run.signal
    );
    for await (const chunk of stream) {
      const err = (chunk as unknown as { error?: { message?: string } }).error;
      if (err) throw new Error(err.message || "The engine reported an error");
      const delta = chunk.choices?.[0]?.delta;
      const thought = delta?.reasoning_content;
      if (thought) {
        if (!thinkStart) thinkStart = Date.now();
        reasoning += thought;
        thinkingMs = Date.now() - thinkStart;
        run.onReasoning?.(reasoning, false, thinkingMs);
      }
      if (delta?.content) {
        if (thinkStart && !thinkDone) {
          thinkDone = true;
          thinkingMs = Date.now() - thinkStart;
          run.onReasoning?.(reasoning, true, thinkingMs);
        }
        content += delta.content;
        run.onContent?.(content);
      }
    }
  } catch (e) {
    if (!run.signal?.aborted) throw e;
  }
  if (thinkStart && !thinkDone) run.onReasoning?.(reasoning, true, Date.now() - thinkStart);
  return { content, reasoning, thinkingMs, aborted: run.signal?.aborted ?? false };
}
