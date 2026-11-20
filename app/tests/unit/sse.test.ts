import { describe, it, expect } from "vitest";
import { parseSseStream } from "../../src/lib/api/sse.ts";

function createMockStream(chunks: string[]): ReadableStreamDefaultReader<Uint8Array> {
  const encoder = new TextEncoder();
  let index = 0;

  const stream = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (index < chunks.length) {
        const chunk = chunks[index];
        index++;
        if (chunk) {
          controller.enqueue(encoder.encode(chunk));
        }
      } else {
        controller.close();
      }
    }
  });

  return stream.getReader();
}

describe("parseSseStream", () => {
  it("parses single data chunk and [DONE]", async () => {
    const rawChunks = ['data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n', "data: [DONE]\n\n"];

    const reader = createMockStream(rawChunks);
    const results = [];

    for await (const chunk of parseSseStream(reader)) {
      results.push(chunk);
    }

    expect(results.length).toBe(1);
    expect(results[0]?.choices?.[0]?.delta?.content).toBe("Hello");
  });

  it("handles split packets across chunks seamlessly", async () => {
    const rawChunks = [
      'data: {"choices":[{"delta":{"con',
      'tent":"World"}}]}\n\n',
      "data: [DONE]\n\n"
    ];

    const reader = createMockStream(rawChunks);
    const results = [];

    for await (const chunk of parseSseStream(reader)) {
      results.push(chunk);
    }

    expect(results.length).toBe(1);
    expect(results[0]?.choices?.[0]?.delta?.content).toBe("World");
  });
});
