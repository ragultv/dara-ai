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

  test("lists models, shows file state, and confirms before deleting", async ({ page }) => {
    await page.goto("http://127.0.0.1:1420/#/models");

    await expect(page.getByRole("heading", { name: "Models", exact: true })).toBeVisible({
      timeout: 20000
    });
    await expect(page.getByText("Disk free:")).toBeVisible();

    await expect(page.getByRole("heading", { name: "Dara Nano" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Dara Core 4B" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Dara Pro 9B" })).toBeVisible();

    await expect(page.getByText("Loaded", { exact: true })).toBeVisible();
    await expect(page.getByText("Image support").first()).toBeVisible();
    await expect(page.getByText("Flash drafter").first()).toBeVisible();

    const enabledDelete = page
      .getByRole("button", { name: "Delete files", exact: true })
      .and(page.locator(":enabled"));
    if ((await enabledDelete.count()) > 0) {
      await enabledDelete.first().click();
      await expect(page.getByRole("dialog", { name: /^Delete / })).toBeVisible();
      await expect(page.getByText(/Disk space freed/)).toBeVisible();
      await page.getByRole("button", { name: "Cancel", exact: true }).click();
      await expect(page.getByRole("dialog")).not.toBeVisible();
    }
  });
});
