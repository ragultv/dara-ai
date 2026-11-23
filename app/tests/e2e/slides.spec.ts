import { test, expect, type Page } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createServer, type ViteDevServer } from "vite";
import { createMockServer } from "../../mocks/server.ts";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Slides, end to end in a browser: make a deck, let Dara write it, edit and arrange slides, change the look, present, and see it in the
// gallery. The app and the mock server run on their own ports and the page's calls to the engine's port are sent to the mock, so this can
// run next to a live engine and never touches its data. The model is a stub: it streams the deck below.
const MOCK_PORT = 11436;
const APP_PORT = 1421;
const APP = `http://127.0.0.1:${APP_PORT}`;

const DECK = [
  "# Dara",
  "",
  "AI that runs on your own GPU",
  "",
  "---",
  "",
  "## Why local",
  "",
  "- Your chats stay private",
  "- No cloud bill",
  "",
  "---",
  "",
  "## How it works",
  "",
  "- Pick a model",
  "- Chat or write",
  "",
  "---",
  "",
  "## Next steps",
  "",
  "- Install Dara",
  "- Try a deck"
].join("\n");

let mockServer: ReturnType<typeof createMockServer>;
let viteServer: ViteDevServer;

test.beforeAll(async () => {
  mockServer = createMockServer({ port: MOCK_PORT });
  await mockServer.start();
  viteServer = await createServer({
    root: path.resolve(__dirname, "../.."),
    configFile: path.resolve(__dirname, "../../vite.config.ts"),
    server: { port: APP_PORT, host: "127.0.0.1", strictPort: true }
  });
  await viteServer.listen();
});

test.afterAll(async () => {
  await viteServer?.close();
  await mockServer?.stop();
});

/** One streamed model answer, in the engine's server-sent-event format. */
function sse(text: string): string {
  const chunk = (delta: object, finish: string | null = null) =>
    `data: ${JSON.stringify({ choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`;
  const parts = text.match(/[\s\S]{1,40}/g) ?? [];
  return (
    chunk({ role: "assistant" }) +
    parts.map((p) => chunk({ content: p })).join("") +
    chunk({}, "stop") +
    "data: [DONE]\n\n"
  );
}

async function openApp(page: Page, answers: string[]) {
  // everything the app sends to the engine's usual port goes to the mock; the model's answers come from this list
  await page.route("http://127.0.0.1:11435/**", async (route) => {
    const req = route.request();
    if (req.url().endsWith("/v1/chat/completions") && req.method() === "POST") {
      const body = answers.shift() ?? "Done.";
      return route.fulfill({
        status: 200,
        headers: { "content-type": "text/event-stream", "access-control-allow-origin": "*" },
        body: sse(body)
      });
    }
    const url = req.url().replace(":11435", `:${MOCK_PORT}`);
    const res = await route.fetch({ url });
    return route.fulfill({ response: res });
  });
  await page.goto(`${APP}/#/artifacts`);
}

const slideButton = (page: Page, n: number) =>
  page
    .getByRole("navigation", { name: "Slides" })
    .getByRole("button", { name: new RegExp(`^Slide ${n}\\b`) });
// the slide rows only: the bullet points inside a thumbnail are list items too
const slideCount = (page: Page) => page.locator('nav[aria-label="Slides"] > ol > li');

test.describe("Slides", () => {
  test.setTimeout(90_000);

  test("make a deck, write it with Dara, edit, arrange, theme, present, and find it in the gallery", async ({
    page
  }) => {
    await openApp(page, [DECK, "I wrote a 4-slide deck about Dara."]);

    // ---- a new deck from the gallery ----
    await expect(page.getByRole("button", { name: /^Slides$/ })).toBeEnabled();
    await page.getByRole("button", { name: /^Slides$/ }).click();
    await expect(page.getByRole("navigation", { name: "Slides" })).toBeVisible();
    await expect(slideCount(page)).toHaveCount(1);
    await expect(page.getByRole("button", { name: "Present" })).toBeDisabled(); // nothing to present yet
    await expect(page.getByText("Describe a deck")).toBeVisible();
    await page.screenshot({ path: test.info().outputPath("1-empty-deck.png") });

    // ---- Dara writes it ----
    await page.getByLabel("Message to Dara").fill("A 4-slide pitch for Dara");
    await page.getByRole("button", { name: "Send" }).click();
    await expect(slideCount(page)).toHaveCount(4, { timeout: 20_000 });
    await expect(page.getByText("I wrote a 4-slide deck about Dara.")).toBeVisible({
      timeout: 20_000
    });
    await expect(page.getByLabel("Deck title")).toHaveValue("Dara");
    await expect(page.getByRole("button", { name: "Present" })).toBeEnabled();
    await page.screenshot({ path: test.info().outputPath("2-written.png") });

    // the title slide is centred and large, the content slides are not
    await slideButton(page, 1).click();
    await expect(page.locator(".slide-title .ProseMirror")).toContainText(
      "AI that runs on your own GPU"
    );
    await slideButton(page, 2).click();
    await expect(page.locator(".slide-content .ProseMirror")).toContainText(
      "Your chats stay private"
    );

    // ---- edit a slide: the list follows what is typed ----
    const editor = page.locator(".slide .ProseMirror");
    await editor.click();
    await page.keyboard.press("Control+End");
    await page.keyboard.type(" and fast");
    await expect(slideButton(page, 2)).toHaveAttribute("aria-label", /Why local/);
    await expect(
      page.getByRole("navigation", { name: "Slides" }).locator(".slide-html").nth(1)
    ).toContainText("No cloud bill and fast");
    await expect(page.getByText("Saved")).toBeVisible({ timeout: 10_000 });

    // ---- add, duplicate, move, delete ----
    await page.getByRole("button", { name: "New slide" }).click();
    await expect(slideCount(page)).toHaveCount(5);
    await expect(slideButton(page, 3)).toHaveAttribute("aria-label", /\(blank\)/); // the new slide is after the one that was open
    await slideButton(page, 3).hover();
    await page.getByRole("button", { name: "Delete slide" }).nth(2).click();
    await expect(slideCount(page)).toHaveCount(4);

    await slideButton(page, 4).hover();
    await page.getByRole("button", { name: "Move up" }).nth(3).click();
    await expect(slideButton(page, 3)).toHaveAttribute("aria-label", /Next steps/);
    await expect(slideButton(page, 4)).toHaveAttribute("aria-label", /How it works/);

    await slideButton(page, 1).hover();
    await page.getByRole("button", { name: "Duplicate slide" }).nth(0).click();
    await expect(slideCount(page)).toHaveCount(5);
    await expect(slideButton(page, 2)).toHaveAttribute("aria-label", /Dara/);
    await page.getByRole("button", { name: "Delete slide" }).nth(1).click(); // back to four

    // ---- the look ----
    await page.getByLabel("Theme").selectOption("midnight");
    await expect(page.locator(".slide-theme-midnight").first()).toBeVisible();
    await page.screenshot({ path: test.info().outputPath("3-midnight.png") });

    // ---- present: it starts on the slide that is open, like Google Slides ----
    const show = page.getByRole("dialog", { name: /Presenting/ });
    await slideButton(page, 3).click();
    await page.getByRole("button", { name: "Present" }).click();
    await expect(show).toBeVisible();
    await expect(show.getByRole("status")).toHaveText("3 / 4");
    await page.keyboard.press("Escape");
    await expect(show).toBeHidden();
    await slideButton(page, 1).click();
    await page.getByRole("button", { name: "Present" }).click();
    await expect(show).toBeVisible();
    await expect(show.getByRole("status")).toHaveText("1 / 4");
    await page.keyboard.press("ArrowRight");
    await expect(show.getByRole("status")).toHaveText("2 / 4");
    await expect(show).toContainText("Why local");
    await page.keyboard.press("Space");
    await page.keyboard.press("ArrowLeft");
    await expect(show.getByRole("status")).toHaveText("2 / 4");
    await page.keyboard.press("End");
    await expect(show.getByRole("status")).toHaveText("4 / 4");
    await page.screenshot({ path: test.info().outputPath("4-present.png") });
    await page.keyboard.press("Escape");
    await expect(show).toBeHidden();

    // ---- it is in the gallery, as its first slide ----
    await page.getByRole("button", { name: "Back to artifacts" }).click();
    await expect(page.getByRole("button", { name: /Open Dara/ })).toBeVisible();
    await expect(
      page.locator(".slide-theme-midnight .slide-html h1", { hasText: "Dara" }).first()
    ).toBeVisible();
    await page.screenshot({ path: test.info().outputPath("5-gallery.png") });

    // ---- and it comes back as it was left ----
    await page.getByRole("button", { name: /Open Dara/ }).click();
    await expect(slideCount(page)).toHaveCount(4);
    await expect(page.getByLabel("Theme")).toHaveValue("midnight");
  });

  test("Dara changes slides by number, adds one before another, and each answer can be undone", async ({
    page
  }) => {
    await openApp(page, [
      DECK,
      "Wrote it.",
      "@replace 2\n## Why local\n\n- Private by default\n- No cloud bill",
      "Changed a slide.",
      "@insert before 4\n## Pricing\n\n- Free to start",
      "Added a slide."
    ]);
    await page.getByRole("button", { name: /^Slides$/ }).click();
    await page.getByLabel("Message to Dara").fill("pitch");
    await page.getByRole("button", { name: "Send" }).click();
    await expect(slideCount(page)).toHaveCount(4, { timeout: 20_000 });
    await expect(page.getByText("Wrote it.")).toBeVisible({ timeout: 20_000 });

    // change slide 2
    await page.getByLabel("Message to Dara").fill("make the first point stronger");
    await page.getByRole("button", { name: "Send" }).click();
    await expect(page.getByText("Changed a slide.")).toBeVisible({ timeout: 20_000 });
    await slideButton(page, 2).click();
    await expect(page.locator(".slide .ProseMirror")).toContainText("Private by default");
    await expect(slideCount(page)).toHaveCount(4);

    // add a slide before the last one: it becomes slide 4 and "Next steps" is pushed to 5
    await page.getByLabel("Message to Dara").fill("add a pricing slide before the last slide");
    await page.getByRole("button", { name: "Send" }).click();
    await expect(page.getByText("Added a slide.")).toBeVisible({ timeout: 20_000 });
    await expect(slideCount(page)).toHaveCount(5);
    await expect(slideButton(page, 4)).toHaveAttribute("aria-label", /Pricing/);
    await expect(slideButton(page, 5)).toHaveAttribute("aria-label", /Next steps/);

    // Undo is for the latest answer only: it takes the added slide away and keeps the change made before it
    await expect(page.getByRole("button", { name: "Undo" })).toHaveCount(1);
    await page.getByRole("button", { name: "Undo" }).click();
    await expect(slideCount(page)).toHaveCount(4);
    await expect(slideButton(page, 4)).toHaveAttribute("aria-label", /Next steps/);
    await slideButton(page, 2).click();
    await expect(page.locator(".slide .ProseMirror")).toContainText("Private by default");
    await expect(page.getByRole("button", { name: "Undo" })).toHaveCount(0);
  });
});
