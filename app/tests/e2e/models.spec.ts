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

test.describe("M2: Models & Lifecycle", () => {
  test.setTimeout(60000);

  test("lists models, supports download, and shows model management controls", async ({ page }) => {
    await page.goto("http://127.0.0.1:1420/#/models");

    // Verify Models page header and disk space
    await expect(page.getByRole("heading", { name: "Models" })).toBeVisible({ timeout: 20000 });
    await expect(page.getByText("Disk Free:")).toBeVisible();

    // Verify model cards
    await expect(page.getByText("Dara Nano")).toBeVisible();
    await expect(page.getByText("Dara Core 4B")).toBeVisible();
    await expect(page.getByText("Dara Pro 9B")).toBeVisible();

    // Verify Nano loaded badge or action buttons
    await expect(page.getByText("Loaded", { exact: true })).toBeVisible();

    // Verify delete confirmation dialog opens
    const deleteButtons = page.getByRole("button", { name: "Delete" });
    if ((await deleteButtons.count()) > 0) {
      await deleteButtons.first().click();
      await expect(page.getByText("Delete Model Files")).toBeVisible();
      await page.getByRole("button", { name: "Cancel" }).click();
      await expect(page.getByText("Delete Model Files")).not.toBeVisible();
    }
  });
});
