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

test.describe("M5: Monitor & Settings", () => {
  test.setTimeout(60000);

  test.beforeEach(async ({ context }) => {
    try {
      await context.grantPermissions(["clipboard-read", "clipboard-write"]);
    } catch {
      // ignore
    }
  });

  test("renders telemetry on Monitor page, verifies cards and sparkline", async ({ page }) => {
    await page.goto("http://127.0.0.1:1420/#/monitor");

    // Header & live indicator
    await expect(page.getByRole("heading", { name: "Hardware & Engine Monitor" })).toBeVisible({
      timeout: 20000
    });
    await expect(page.getByText("Live", { exact: true })).toBeVisible();

    // GPU Telemetry Card
    await expect(page.getByRole("heading", { name: "GPU Telemetry" })).toBeVisible();
    await expect(page.getByText("NVIDIA GeForce GTX 1650")).toBeVisible();
    await expect(page.getByText("VRAM Allocation", { exact: true })).toBeVisible();
    await expect(page.getByText("50°C")).toBeVisible();

    // Inference Speed & Sparkline
    await expect(page.getByRole("heading", { name: "Inference Speed" })).toBeVisible();
    await expect(page.getByLabel("Live generation TPS sparkline")).toBeVisible();

    // Model Placement Card
    await expect(page.getByRole("heading", { name: "Model Placement" })).toBeVisible();
    await expect(page.getByText("35 / 35 on GPU")).toBeVisible();

    // Storage & System Card
    await expect(page.getByRole("heading", { name: "Storage & System" })).toBeVisible();
    await expect(page.getByText("Volume Free Space", { exact: true })).toBeVisible();

    // Copy Diagnostics button exists and is clickable
    const copyBtn = page.getByRole("button", { name: "Copy diagnostics" });
    await expect(copyBtn).toBeVisible();
    await copyBtn.click();
    // Toast appears
    await expect(page.getByText(/Diagnostics copied/i)).toBeVisible();
  });

  test("manages settings, toggles network exposure with API key, and restores defaults", async ({
    page
  }) => {
    await page.goto("http://127.0.0.1:1420/#/settings");

    // Header
    await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible({ timeout: 20000 });

    // Appearance theme buttons
    await expect(page.getByRole("button", { name: "Dark" })).toBeVisible();
    await page.getByRole("button", { name: "Dark" }).click();

    // Network exposure switch
    const networkToggle = page.getByLabel("Allow other devices on my network");
    await expect(networkToggle).toBeVisible();
    await networkToggle.click();

    // Warning appears
    await expect(page.getByText("Network Security Warning")).toBeVisible();

    // API key controls appear
    await expect(page.getByText("Engine API Key")).toBeVisible();
    await expect(page.getByRole("button", { name: "Reveal Key" })).toBeVisible();
    await page.getByRole("button", { name: "Reveal Key" }).click();
    await expect(page.getByRole("button", { name: "Hide Key" })).toBeVisible();
    await expect(page.getByText(/dara-key-/)).toBeVisible();

    // Reset to Defaults
    const resetBtn = page.getByRole("button", { name: "Reset to Defaults" });
    await expect(resetBtn).toBeVisible();
    await resetBtn.click();
    await expect(page.getByText("Settings restored to factory defaults")).toBeVisible();
  });
});
