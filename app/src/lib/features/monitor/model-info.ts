// The `model` block of /dara/stats is an open record; these readers narrow single fields without guessing.

export type ModelRecord = Record<string, unknown>;

export function readString(model: ModelRecord | undefined, key: string): string | undefined {
  const v = model?.[key];
  return typeof v === "string" && v.length > 0 ? v : undefined;
}

export function readNumber(model: ModelRecord | undefined, key: string): number | undefined {
  const v = model?.[key];
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}

export function readBool(model: ModelRecord | undefined, key: string): boolean | undefined {
  const v = model?.[key];
  return typeof v === "boolean" ? v : undefined;
}

/** Flash is on when the model info names a drafter file; the server sends an empty string otherwise. */
export function isFlashLoaded(model: ModelRecord | undefined): boolean {
  return readString(model, "flash") !== undefined;
}
