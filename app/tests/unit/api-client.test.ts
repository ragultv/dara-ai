import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { DaraApiClient } from "../../src/lib/api/client.ts";
import { ApiError } from "../../src/lib/api/errors.ts";
import { createMockServer } from "../../mocks/server.ts";

describe("DaraApiClient with mock server", () => {
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

  it("fetches health status and returns ok with model", async () => {
    const health = await client.getHealth();
    expect(health.status).toBe("ok");
    expect(health.model).toBe("dara-nano");
    expect(health.busy).toBe(false);
  });

  it("fetches model list", async () => {
    const models = await client.getModels();
    expect(models.object).toBe("list");
    expect(Array.isArray(models.data)).toBe(true);
    expect(models.data.length).toBeGreaterThan(0);
  });

  it("handles 404 errors as ApiError with typed status", async () => {
    await expect(client.request("/non-existent-endpoint")).rejects.toThrow(ApiError);
    try {
      await client.request("/non-existent-endpoint");
    } catch (err: unknown) {
      expect(err).toBeInstanceOf(ApiError);
      const apiErr = err as ApiError;
      expect(apiErr.status).toBe(404);
      expect(apiErr.errorType).toBe("not_found");
    }
  });

  it("includes Authorization Bearer header when configured", async () => {
    const authedMock = createMockServer({ apiKey: "test-secret-key" });
    const port = await authedMock.start();

    // Client without key gets 401
    const unauthedClient = new DaraApiClient({ baseUrl: `http://127.0.0.1:${port}` });
    await expect(unauthedClient.getHealth()).rejects.toThrow(ApiError);

    // Client with valid key succeeds
    const authedClient = new DaraApiClient({
      baseUrl: `http://127.0.0.1:${port}`,
      apiKey: "test-secret-key"
    });
    const health = await authedClient.getHealth();
    expect(health.status).toBe("ok");

    await authedMock.stop();
  });
});
