/**
 * A short title for the model's thinking, taken from its own words (nothing is invented): the latest step heading
 * ("1. **Analyze the Request:** ..." gives "Analyze the Request"), else the first sentence, else a plain label.
 */
export function thinkingTitle(text: string, fallback = "Thinking"): string {
  const stripped = text.replace(/\r/g, "");
  // the last bold step label, e.g. "**Analyze the Request:**" or "**Plan**"
  let last: string | null = null;
  for (const m of stripped.matchAll(/\*\*([^*\n]{3,70}?)\*\*/g)) {
    const t = m[1]!.replace(/[:：]\s*$/, "").trim();
    if (t && !/^(thinking process|thought process)$/i.test(t)) last = t;
  }
  if (last) return last;

  // a markdown heading
  const heading = [...stripped.matchAll(/^#{1,4}\s+(.{3,70})$/gm)].pop();
  if (heading) return heading[1]!.replace(/[*_`]/g, "").trim();

  // first sentence of the first real line
  const first = stripped
    .split("\n")
    .map((l) => l.replace(/^[\s>*\-\d.)]+/, "").trim())
    .find((l) => l.length > 3 && !/^(thinking process|thought process):?$/i.test(l));
  if (!first) return fallback;
  const sentence = first
    .split(/(?<=[.!?:])\s/)[0]!
    .replace(/[*_`]/g, "")
    .replace(/[:.]$/, "");
  return sentence.length > 60 ? `${sentence.slice(0, 57).trimEnd()}...` : sentence;
}
