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

test.describe("M4: Model Options & Hardware Benchmarks", () => {
  test.setTimeout(60000);

  test("navigates to options, verifies GPU specs, format cards, context ladder, and runs benchmark", async ({
    page
  }) => {
    // 1. Start on Models page
    await page.goto("http://127.0.0.1:1420/#/models");

    // Wait for models to load and click Options on Dara Nano
    const optionsBtn = page.getByRole("button", { name: "Options" }).first();
    await expect(optionsBtn).toBeVisible({ timeout: 20000 });
    await optionsBtn.click();

    // 2. Verify Options screen renders with GPU and format cards
    await expect(page.getByRole("heading", { name: "Dara Nano Options" })).toBeVisible({
      timeout: 10000
    });
    await expect(page.getByText("NVIDIA GeForce GTX 1650")).toBeVisible();
    await expect(page.getByText("INT8 (default)", { exact: true })).toBeVisible();
    await expect(page.getByText("INT8 keys + 4-bit values", { exact: true })).toBeVisible();

    // 3. Verify Context Ladder table renders
    await expect(page.getByText("Context Ladder")).toBeVisible();
    await expect(page.getByText("100% on GPU").first()).toBeVisible();

    // 4. Test format selection switch
    await page.getByText("INT8 keys + 4-bit values", { exact: true }).click();

    // 5. Open Benchmark dialog
    const measureBtn = page.getByRole("button", { name: "Measure Benchmarks on this PC" });
    await expect(measureBtn).toBeVisible();
    await measureBtn.click();

    await expect(
      page.getByRole("heading", { name: "Run Hardware Inference Benchmark" })
    ).toBeVisible();
    await expect(page.getByRole("button", { name: "Start Benchmark" })).toBeVisible();

    // Start benchmark
    await page.getByRole("button", { name: "Start Benchmark" }).click();
    await expect(page.getByText(/Status: (RUNNING|DONE)/i)).toBeVisible();

    // Close dialog via Cancel Benchmark or Close
    const cancelBtn = page.getByRole("button", { name: "Cancel Benchmark" });
    if (await cancelBtn.isVisible()) {
      await cancelBtn.click();
    } else {
      await page.getByRole("button", { name: "Close" }).click();
    }
    await expect(
      page.getByRole("heading", { name: "Run Hardware Inference Benchmark" })
    ).not.toBeVisible();

    // 6. Test "Load with these settings"
    const loadSettingsBtn = page.getByRole("button", { name: "Load with these settings" });
    await expect(loadSettingsBtn).toBeVisible();
    await loadSettingsBtn.click();

    // Verify toast confirmation and redirection to chat
    await expect(page.getByText(/Loaded dara-nano with/i)).toBeVisible();
  });
});
