// From assistant-ui's range helpers (elements-range): keeps a count that arrives negative, NaN or past the end of a list inside 0..length.

/** Constrains a value to min..max. NaN maps to min; an empty collection can invert the bounds and max then wins. */
export function clamp(value: number, min: number, max: number): number {
  if (Number.isNaN(value)) return min;
  return Math.min(max, Math.max(min, value));
}

/** The first `count` items, for a `count` that may be out of range. */
export function take<T>(items: readonly T[], count: number): T[] {
  return items.slice(0, Math.floor(clamp(count, 0, items.length)));
}
