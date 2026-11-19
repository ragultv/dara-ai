import { describe, it, expect, beforeEach } from "vitest";
import { themeStore } from "../../src/lib/stores/theme.svelte.ts";

describe("ThemeStore", () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.removeAttribute("data-theme");
  });

  it("initializes with default system mode", () => {
    themeStore.setMode("system");
    expect(themeStore.mode).toBe("system");
    expect(document.documentElement.getAttribute("data-theme")).toBeNull();
  });

  it("sets and persists light theme", () => {
    themeStore.setMode("light");
    expect(themeStore.mode).toBe("light");
    expect(themeStore.resolved).toBe("light");
    expect(localStorage.getItem("dara-theme")).toBe("light");
    expect(document.documentElement.getAttribute("data-theme")).toBe("light");
  });

  it("sets and persists dark theme", () => {
    themeStore.setMode("dark");
    expect(themeStore.mode).toBe("dark");
    expect(themeStore.resolved).toBe("dark");
    expect(localStorage.getItem("dara-theme")).toBe("dark");
    expect(document.documentElement.getAttribute("data-theme")).toBe("dark");
  });
});
