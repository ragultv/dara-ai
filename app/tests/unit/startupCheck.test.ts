import { describe, expect, it } from "vitest";
import { parseCapability, startupProblem } from "../../src/lib/features/models/startupCheck.ts";
import type { Stats } from "../../src/lib/api/types.ts";

const stats = (gpu?: Partial<NonNullable<Stats["gpu"]>>): Stats =>
  ({
    uptime_seconds: 1,
    pid: 1,
    queued: 0,
    context: null,
    ...(gpu ? { gpu } : {})
  }) as unknown as Stats;

describe("first-launch check", () => {
  it("waits for an engine that is still starting, then says it did not start", () => {
    expect(startupProblem({ engineUp: false, stats: null, waitedMs: 3000 })).toBeNull();
    expect(startupProblem({ engineUp: false, stats: null, waitedMs: 20000 })?.kind).toBe("engine");
  });

  it("says when there is no NVIDIA graphics card, with where to get a driver", () => {
    const p = startupProblem({ engineUp: true, stats: stats(), waitedMs: 0 });
    expect(p?.kind).toBe("no_gpu");
    expect(p?.helpUrl).toMatch(/nvidia\.com/);
    expect(
      startupProblem({ engineUp: true, stats: stats({ name: "  " }), waitedMs: 0 })?.kind
    ).toBe("no_gpu");
  });

  it("says when the card is older than Turing, naming it", () => {
    const p = startupProblem({
      engineUp: true,
      stats: stats({ name: "GeForce GTX 1060", compute_capability: "6.1" }),
      waitedMs: 0
    });
    expect(p?.kind).toBe("old_gpu");
    expect(p?.title).toContain("GTX 1060");
  });

  it("says nothing when the card is fine, or when the numbers are not known", () => {
    expect(
      startupProblem({
        engineUp: true,
        stats: stats({ name: "RTX 4060", compute_capability: "8.9" }),
        waitedMs: 0
      })
    ).toBeNull();
    expect(
      startupProblem({
        engineUp: true,
        stats: stats({ name: "GTX 1650", compute_capability: "7.5" }),
        waitedMs: 0
      })
    ).toBeNull();
    expect(startupProblem({ engineUp: true, stats: stats({ name: "X" }), waitedMs: 0 })).toBeNull();
    expect(startupProblem({ engineUp: true, stats: null, waitedMs: 0 })).toBeNull();
  });

  it("reads the capability the engine reports", () => {
    expect(parseCapability("8.6")).toBe(8.6);
    expect(parseCapability(undefined)).toBeNull();
    expect(parseCapability("n/a")).toBeNull();
  });
});
