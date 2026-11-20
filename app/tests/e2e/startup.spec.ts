import { test, expect } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createServer, type ViteDevServer } from "vite";
import { createMockServer } from "../../mocks/server.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let mockServer: ReturnType<typeof createMockServer>;
let viteServer: ViteDevServer;

test.beforeAll(async () => {
  mockServer = createMockServer({ port: 11435 });
  await mockServer.start();

  const appRoot = path.resolve(__dirname, "../..");
  viteServer = await createServer({
    root: appRoot,
    configFile: path.resolve(appRoot, "vite.config.ts"),
    server: { port: 1420, host: "127.0.0.1" }
  });
  await viteServer.listen();
});

test.afterAll(async () => {
  if (viteServer) {
    await viteServer.close();
  }
  if (mockServer) {
    await mockServer.stop();
  }
});

test.describe("Startup and Health Flow", () => {
  test("shows health status and loaded model name from mock server", async ({ page }) => {
    await page.goto("http://127.0.0.1:1420");

    // Verify engine ready view
    await expect(page.getByText("Engine is ready")).toBeVisible({ timeout: 20000 });
    await expect(page.getByText("dara-nano")).toBeVisible();
    await expect(page.getByText("● Active")).toBeVisible();
    await expect(page.getByText("Endpoint")).toBeVisible();
  });
});
