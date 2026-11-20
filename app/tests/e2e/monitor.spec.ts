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

  test("renders real telemetry on the Monitor page and hides nothing it was given", async ({
    page
  }) => {
    await page.goto("http://127.0.0.1:1420/#/monitor");

    await expect(page.getByRole("heading", { name: "Monitor", level: 1 })).toBeVisible({
      timeout: 20000
    });
    await expect(page.getByText("Live", { exact: true })).toBeVisible();

    // Summary tiles come from the recorded sample (dara-nano, q8, 35 layers on the GPU)
    await expect(page.getByText("dara-nano")).toBeVisible();
    await expect(page.getByText("35 / 35")).toBeVisible();

    // GPU card: name and temperature are real values from the sample
    await expect(page.getByRole("heading", { name: "GPU", exact: true })).toBeVisible();
    await expect(page.getByText("NVIDIA GeForce GTX 1650")).toBeVisible();
    await expect(page.getByText("50 °C")).toBeVisible();

    // Idle engine: the decode chart shows an idle state, not a fake line
    await expect(page.getByText("Appears while the engine generates")).toBeVisible();

    await expect(page.getByRole("heading", { name: "Disk" })).toBeVisible();
    await expect(
      page.getByText("No request has been served since the engine started.")
    ).toBeVisible();

    const copyBtn = page.getByRole("button", { name: "Copy diagnostics" });
    await expect(copyBtn).toBeVisible();
    await copyBtn.click();
    await expect(page.getByText(/Diagnostics copied/i)).toBeVisible();
  });

  test("Settings: section navigation, theme, and no fake engine controls", async ({ page }) => {
    await page.goto("http://127.0.0.1:1420/#/settings");

    await expect(page.getByRole("heading", { name: "Settings", level: 1 })).toBeVisible({
      timeout: 20000
    });

    const nav = page.getByRole("navigation", { name: "Settings sections" });
    for (const label of ["General", "Engine", "Defaults", "Data", "About"]) {
      await expect(nav.getByRole("button", { name: label })).toBeVisible();
    }

    await page.getByRole("radio", { name: "Dark" }).click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");

    // Outside the desktop app the engine settings are disabled with an explanation
    await expect(page.getByText(/running outside the app/i)).toBeVisible();
    // The fake key and the Regenerate button are gone
    await expect(page.getByRole("button", { name: "Regenerate" })).toHaveCount(0);
    await expect(page.getByText("dara_sk_live")).toHaveCount(0);
  });
});
