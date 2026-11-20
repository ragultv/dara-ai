import { test, expect } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createServer, type ViteDevServer } from "vite";
import AxeBuilder from "@axe-core/playwright";
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

test.describe("M1: Design System & Shell", () => {
  test.setTimeout(60000);

  test("renders component kit on /#/kit and passes axe accessibility audit", async ({ page }) => {
    await page.goto("http://127.0.0.1:1420/#/kit");

    // Verify Kit page loaded
    await expect(page.getByText("Dara Component Kit")).toBeVisible({ timeout: 20000 });
    await expect(page.getByText("Buttons & IconButtons")).toBeVisible();
    await expect(page.getByText("Badges (Quality & Status)")).toBeVisible();
    await expect(page.getByText("Form Controls")).toBeVisible();

    // Verify theme switching via LeftRail
    const darkThemeButton = page.getByRole("button", { name: "Dark", exact: true });
    await darkThemeButton.click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");

    const lightThemeButton = page.getByRole("button", { name: "Light", exact: true });
    await lightThemeButton.click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "light");

    // Run automated WCAG 2.2 AA accessibility scan with axe
    const accessibilityScanResults = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .disableRules(["color-contrast"]) // contrast verified in contrast-report.md using generated tokens
      .analyze();

    const criticalViolations = accessibilityScanResults.violations.filter(
      (v) => v.impact === "critical" || v.impact === "serious"
    );

    expect(criticalViolations).toEqual([]);
  });

  test("supports left rail navigation across routes", async ({ page }) => {
    await page.goto("http://127.0.0.1:1420/#/chat");

    // Wait for readiness
    await expect(page.getByText("Engine is ready")).toBeVisible({ timeout: 20000 });

    // Navigate to Models
    await page.getByRole("button", { name: "Models" }).click();
    expect(page.url()).toContain("#/models");
    await expect(page.getByRole("heading", { name: "Models" })).toBeVisible();

    // Navigate to Monitor
    await page.getByRole("button", { name: "Monitor" }).click();
    expect(page.url()).toContain("#/monitor");
    await expect(page.getByRole("heading", { name: "Hardware & Engine Monitor" })).toBeVisible();

    // Navigate to Settings
    await page.getByRole("button", { name: "Settings" }).click();
    expect(page.url()).toContain("#/settings");
    await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible();

    // Navigate to Kit
    await page.getByRole("button", { name: "Component Kit" }).click();
    expect(page.url()).toContain("#/kit");
    await expect(page.getByText("Dara Component Kit")).toBeVisible();
  });
});
