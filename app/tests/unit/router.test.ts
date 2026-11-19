import { describe, it, expect, beforeEach } from "vitest";
import { router } from "../../src/lib/router/router.svelte.ts";

describe("Router", () => {
  beforeEach(() => {
    window.location.hash = "";
  });

  it("defaults to chat route", () => {
    router.navigate("chat");
    expect(router.current).toBe("chat");
    expect(window.location.hash).toBe("#/");
  });

  it("navigates to models route", () => {
    router.navigate("models");
    expect(router.current).toBe("models");
    expect(window.location.hash).toBe("#/models");
  });

  it("navigates to kit route", () => {
    router.navigate("kit");
    expect(router.current).toBe("kit");
    expect(window.location.hash).toBe("#/kit");
  });

  it("navigates to settings and monitor routes", () => {
    router.navigate("settings");
    expect(router.current).toBe("settings");
    expect(window.location.hash).toBe("#/settings");

    router.navigate("monitor");
    expect(router.current).toBe("monitor");
    expect(window.location.hash).toBe("#/monitor");
  });
});
