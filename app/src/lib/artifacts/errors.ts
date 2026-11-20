import { ApiError } from "../api/errors.ts";

/**
 * A message for a failed call about comments, chat, images or history. An engine older than the app answers 404 for these routes,
 * which is not "the document is missing" but "restart Dara so the engine is updated".
 */
export function collabMessage(e: unknown, what: string): string {
  if (e instanceof ApiError) {
    if (e.status === 404 || e.status === 405)
      return `This engine is older than the app and cannot keep ${what} yet. Close and reopen Dara to update it.`;
    if (e.status === 0)
      return `The engine did not accept that (${what}). If Dara was just updated, close and reopen it so the engine is updated too.`;
    if (e.message) return e.message;
  }
  return e instanceof Error && e.message ? e.message : `Something went wrong with ${what}.`;
}

/** True when the failure is the older-engine case above. */
export function isOldEngine(e: unknown): boolean {
  return e instanceof ApiError && (e.status === 404 || e.status === 405);
}
