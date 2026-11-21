import { apiClient } from "../api/client.ts";
import type { ChatMessage } from "../api/types.ts";

// "Ask Dara" in a document: the user types an instruction, the local model returns the whole updated document in Markdown, and the editor
// shows it as it streams. The model never sees anything but the document and the instruction.

export const EDIT_SYSTEM_PROMPT = [
  "You are Dara, the writing assistant inside a document editor.",
  "The user gives you the current document in Markdown and one instruction.",
  "Reply with the complete updated document in Markdown and nothing else: no greeting, no explanation, no notes after it, and do not wrap the document in a code fence.",
  "Change only what the instruction asks for and keep every other part word for word.",
  "Use plain Markdown: # headings, **bold**, - bullet lists, 1. numbered lists, - [ ] task lists, | tables |, > quotes and fenced code blocks.",
  "If the document is empty, write it from the instruction."
].join(" ");

export function buildEditMessages(markdown: string, instruction: string): ChatMessage[] {
  const doc = markdown.trim() ? markdown : "(the document is empty)";
  return [
    { role: "system", content: EDIT_SYSTEM_PROMPT },
    {
      role: "user",
      content: `Current document:\n<document>\n${doc}\n</document>\n\nInstruction: ${instruction.trim()}`
    }
  ];
}

/**
 * Models sometimes wrap the whole answer in a ```markdown fence or open with "Here is the updated document:". Strip those, never anything inside.
 * Works on partial text too, so the editor can show a clean preview while it streams.
 */
export function cleanEditResult(text: string): string {
  let t = text.replace(/\r\n/g, "\n");
  t = t.replace(/^\s*<document>\s*\n?/i, "").replace(/\n?\s*<\/document>\s*$/i, "");
  // a single fence around everything: ```markdown\n ... \n```
  const open = /^\s*```(?:markdown|md)?[ \t]*\n/i.exec(t);
  if (open) {
    t = t.slice(open[0].length);
    t = t.replace(/\n?```[ \t]*\s*$/, "");
  }
  // a one-line lead-in such as "Here is the updated document:"
  t = t.replace(/^\s*(?:here(?:'s| is)[^\n]{0,80}document[^\n]{0,40}:)\s*\n+/i, "");
  return t.replace(/^\n+/, "");
}

/** Room to write the document back plus a margin for growth. About 3 characters per token is deliberately pessimistic. */
export function editMaxTokens(markdown: string): number {
  return Math.min(8192, Math.max(1024, Math.ceil((markdown.length / 3) * 1.4) + 768));
}

export interface EditRun {
  markdown: string;
  instruction: string;
  signal?: AbortSignal;
  /** called with the cleaned text so far, as it streams */
  onText: (cleaned: string) => void;
}

/** Streams the edit; resolves with the final cleaned document. Throws on a network or engine error (an abort is a normal stop: it resolves with what arrived). */
export async function runEdit(run: EditRun): Promise<string> {
  let raw = "";
  try {
    const stream = await apiClient.streamChat(
      {
        messages: buildEditMessages(run.markdown, run.instruction),
        temperature: 0.3,
        max_tokens: editMaxTokens(run.markdown)
      },
      run.signal
    );
    for await (const chunk of stream) {
      const err = (chunk as unknown as { error?: { message?: string } }).error;
      if (err) throw new Error(err.message || "The engine reported an error");
      const delta = chunk.choices?.[0]?.delta?.content;
      if (delta) {
        raw += delta;
        run.onText(cleanEditResult(raw));
      }
    }
  } catch (e) {
    if (!(run.signal?.aborted && raw)) {
      if (run.signal?.aborted) return run.markdown;
      throw e;
    }
  }
  return cleanEditResult(raw);
}
