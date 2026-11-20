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

  test("opens options from a model card, shows the server's numbers, measures and applies", async ({
    page
  }) => {
    await page.goto("http://127.0.0.1:1420/#/models");

    const optionsBtn = page.getByRole("button", { name: "Options", exact: true }).first();
    await expect(optionsBtn).toBeVisible({ timeout: 20000 });
    await optionsBtn.click();

    await expect(page.getByRole("heading", { name: "Dara Nano options" })).toBeVisible({
      timeout: 10000
    });
    await expect(page.getByText(/NVIDIA GeForce GTX 1650/)).toBeVisible();
    await expect(page.getByRole("radio", { name: /INT8 \(default\)/ })).toBeVisible();
    await expect(page.getByRole("radio", { name: /INT8 keys \+ 4-bit values/ })).toBeVisible();

    await expect(page.getByText("Context ladder")).toBeVisible();
    await expect(page.getByText("Fully on the GPU").first()).toBeVisible();
    await expect(page.getByText("Not measured yet").first()).toBeVisible();

    await page.getByRole("radio", { name: /INT8 keys \+ 4-bit values/ }).click();

    await page.getByRole("button", { name: "Measure on this PC" }).click();
    await expect(page.getByRole("dialog", { name: /Measure Dara Nano on this PC/ })).toBeVisible();
    await page.getByRole("button", { name: "Start measuring" }).click();
    await expect(page.getByText(/Measuring\.|Measurement finished/)).toBeVisible();

    await page.getByRole("button", { name: "Load with these settings" }).click();
    await expect(page.getByText(/is loaded with your settings/)).toBeVisible({ timeout: 20000 });
  });

  test("is reachable from the left rail", async ({ page }) => {
    await page.goto("http://127.0.0.1:1420/#/models");
    await page
      .getByRole("navigation", { name: "Main" })
      .getByRole("button", { name: "Options" })
      .click();
    await expect(page.getByRole("heading", { name: /options$/ })).toBeVisible({ timeout: 10000 });
  });
});
