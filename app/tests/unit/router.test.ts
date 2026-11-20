import { describe, it, expect, beforeEach } from "vitest";
import { useRouter } from "../../src/lib/router/useRouter.ts";

describe("Router", () => {
  beforeEach(() => {
    window.location.hash = "";
  });

  it("defaults to chat route", () => {
    useRouter.getState().navigate("chat");
    expect(useRouter.getState().current).toBe("chat");
    expect(window.location.hash).toBe("#/");
  });

  it("navigates to models route", () => {
    useRouter.getState().navigate("models");
    expect(useRouter.getState().current).toBe("models");
    expect(window.location.hash).toBe("#/models");
  });

  it("navigates to kit route", () => {
    useRouter.getState().navigate("kit");
    expect(useRouter.getState().current).toBe("kit");
    expect(window.location.hash).toBe("#/kit");
  });

  it("navigates to settings and monitor routes", () => {
    useRouter.getState().navigate("settings");
    expect(useRouter.getState().current).toBe("settings");
    expect(window.location.hash).toBe("#/settings");

    useRouter.getState().navigate("monitor");
    expect(useRouter.getState().current).toBe("monitor");
    expect(window.location.hash).toBe("#/monitor");
  });
});
