import { describe, expect, it } from "vitest";
import {
  MAX_NUDGES,
  nudgeText,
  promisesMore,
  unfinishedWork
} from "../../src/lib/chat/tools/jobs.ts";
import { WAIT_TOOL, type ToolResultItem } from "../../src/lib/chat/tools/types.ts";

const ok = (text: string, name = "any__tool"): ToolResultItem => ({
  id: "m:0",
  name,
  status: "ok",
  text
});

describe("unfinished work (any connector)", () => {
  it("sees a job that says pending, queued or in progress", () => {
    for (const t of [
      '{"job": {"id": "x1", "status": "pending"}}',
      "status: queued",
      '{"state":"in_progress"}',
      "Job status = processing",
      "The file is still being generated."
    ])
      expect(unfinishedWork([ok(t)]), t).not.toBeNull();
  });

  it("sees a tool saying to check back, with how long", () => {
    expect(
      unfinishedWork([
        ok("Job started. Please wait at least 15 seconds before checking the status again.")
      ])
    ).toEqual({ seconds: 15 });
    expect(unfinishedWork([ok("Not ready. Retry after 30 seconds.")])).toEqual({ seconds: 30 });
    expect(unfinishedWork([ok('{"status":"queued","retry_after_ms": 4000}')])).toEqual({
      seconds: 4
    });
    expect(unfinishedWork([ok("Not ready yet, check back later")])).toEqual({ seconds: null });
  });

  it("caps the wait it asks for and takes the longest of several results", () => {
    expect(unfinishedWork([ok("check again in 500 seconds")])?.seconds).toBe(60);
    expect(
      unfinishedWork([
        ok("wait 5 seconds, status: pending"),
        ok("wait 20 seconds, status: pending")
      ])?.seconds
    ).toBe(20);
  });

  it("sees a result that needs one more call", () => {
    expect(
      unfinishedWork([ok("Pick a candidate. To finish, call the export tool with the chosen id.")])
    ).not.toBeNull();
    expect(unfinishedWork([ok("Next step: confirm the draft")])).not.toBeNull();
  });

  it("leaves finished results, failures and the wait tool alone", () => {
    expect(
      unfinishedWork([ok('{"status": "completed", "url": "https://x.example/d/1"}')])
    ).toBeNull();
    expect(unfinishedWork([ok("2 events (today): 09:30 Stand-up")])).toBeNull();
    expect(
      unfinishedWork([{ id: "m:0", name: "x", status: "error", text: "status: pending" }])
    ).toBeNull();
    expect(
      unfinishedWork([{ id: "m:0", name: "x", status: "denied", text: "check again" }])
    ).toBeNull();
    expect(unfinishedWork([ok("Waited 15 seconds. check again", WAIT_TOOL)])).toBeNull();
    expect(unfinishedWork([])).toBeNull();
  });
});

describe("a reply that gives up on unfinished work", () => {
  it("catches promises to come back", () => {
    for (const r of [
      "I will check back shortly to see if the design is ready!",
      "I'll check the status again in a moment.",
      "Let me check again.",
      "Please wait at least 15 seconds before checking the status again.",
      "Once it's ready I will send you the link."
    ])
      expect(promisesMore(r), r).toBe(true);
  });

  it("leaves a finished answer alone", () => {
    for (const r of [
      "You have a stand-up at 09:30 and a design review at 14:00.",
      "Here is your design: https://example.com/design/1",
      "I could not do that: the tool said access is not allowed."
    ])
      expect(promisesMore(r), r).toBe(false);
  });
});

describe("the nudge", () => {
  it("tells the model to wait for the time the tool gave and check, with the wait tool's name", () => {
    expect(nudgeText({ seconds: 20 })).toContain(`${WAIT_TOOL} block for 20 seconds`);
    expect(nudgeText(null)).toContain("15 seconds");
    expect(MAX_NUDGES).toBeGreaterThan(0);
  });
});
