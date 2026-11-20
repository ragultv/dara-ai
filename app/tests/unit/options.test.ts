import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { DaraApiClient } from "../../src/lib/api/client.ts";
import { createMockServer } from "../../mocks/server.ts";

describe("Model Options & Benchmark API", () => {
  let mockServer: ReturnType<typeof createMockServer>;
  let mockPort: number;
  let client: DaraApiClient;

  beforeAll(async () => {
    mockServer = createMockServer();
    mockPort = await mockServer.start();
    client = new DaraApiClient({ baseUrl: `http://127.0.0.1:${mockPort}` });
  });

  afterAll(async () => {
    await mockServer.stop();
  });

  it("fetches options for dara-nano with formats and recommended profile", async () => {
    const options = await client.getModelOptions("dara-nano");
    expect(options.model).toBe("dara-nano");
    expect(options.formats.length).toBeGreaterThan(0);
    expect(options.formats.some((f) => f.id === "q8")).toBe(true);
    expect(options.recommended).toBeDefined();
    expect(options.recommended.kv).toBe("q8");
    expect(options.machine).toBeDefined();
    expect(options.machine.gpu).toBeDefined();
    expect(options.options.length).toBeGreaterThan(0);
  });

  it("starts a benchmark job, polls progress, and handles job lifecycle", async () => {
    const job = await client.startBenchmark("dara-nano");
    expect(job.id).toBeDefined();
    expect(job.status).toBe("running");

    const polled = await client.getBenchmarkJob(job.id);
    expect(polled.id).toBe(job.id);
    expect(polled.status).toBe("done");

    await expect(client.cancelBenchmark(job.id)).resolves.toBeUndefined();
  });
});
