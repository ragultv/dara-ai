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

test.describe("M3: Chat & Streaming", () => {
  test.setTimeout(60000);

  test("sends message and renders streaming response from mock server", async ({ page }) => {
    await page.goto("http://127.0.0.1:1420/#/chat");

    // Wait for readiness
    await expect(page.getByText("Engine is ready")).toBeVisible({ timeout: 20000 });

    // Find textarea composer
    const textarea = page.getByPlaceholder("Send a message (Shift+Enter for newline)...");
    await expect(textarea).toBeVisible();

    // Type a prompt
    await textarea.fill("Hello Dara");
    await page.getByRole("button", { name: "Send" }).click();

    // User message should appear in chat thread
    await expect(page.getByRole("log").getByText("Hello Dara")).toBeVisible();

    // Assistant response streamed from docs/ui/samples/chat-stream.sse
    await expect(page.getByText("Hi! How can I help you today")).toBeVisible({ timeout: 15000 });
  });

  test("verifies XSS test payloads render completely inert", async ({ page }) => {
    await page.goto("http://127.0.0.1:1420/#/chat");
    await expect(page.getByText("Engine is ready")).toBeVisible({ timeout: 20000 });

    // Listen for alert dialogs
    let alertTriggered = false;
    page.on("dialog", () => {
      alertTriggered = true;
    });

    const textarea = page.getByPlaceholder("Send a message (Shift+Enter for newline)...");
    await textarea.fill('<script>alert("xss")</script><img src="x" onerror="alert(1)">');
    await page.getByRole("button", { name: "Send" }).click();

    await page.waitForTimeout(1000);
    expect(alertTriggered).toBe(false);
  });
});
