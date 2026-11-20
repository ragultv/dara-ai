import type { ChatChunk } from "./types.ts";

export async function* parseSseStream(
  reader: ReadableStreamDefaultReader<Uint8Array>
): AsyncGenerator<ChatChunk, void, unknown> {
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      // Keep unfinished line fragment in buffer
      buffer = lines.pop() ?? "";

      for (const rawLine of lines) {
        const line = rawLine.trim();
        if (!line || !line.startsWith("data:")) continue;

        const dataStr = line.replace(/^data:\s*/, "");
        if (dataStr === "[DONE]") {
          return;
        }

        try {
          const parsed = JSON.parse(dataStr) as ChatChunk;
          yield parsed;
        } catch {
          // Tolerate non-JSON SSE lines
        }
      }
    }

    // Flush any remaining buffer if it starts with data:
    if (buffer.trim().startsWith("data:")) {
      const dataStr = buffer.trim().replace(/^data:\s*/, "");
      if (dataStr !== "[DONE]") {
        try {
          const parsed = JSON.parse(dataStr) as ChatChunk;
          yield parsed;
        } catch {
          // ignore
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
