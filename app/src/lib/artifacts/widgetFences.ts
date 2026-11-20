// Models rarely use the exact fence language they are told to. When the user asked for something interactive, a code block that is clearly
// a small web page (it has a script or form controls) is turned into a widget so it runs instead of sitting there as code. And while a
// widget is still being written, the page shows progress instead of half a page of code.

const FENCE = /^([ \t]{0,3})(`{3,}|~{3,})([^\n]*)\n([\s\S]*?)\n[ \t]{0,3}\2[ \t]*$/gm;

/** A code block that is a runnable page: html (or no language) with a script or form controls in it. */
function looksLikeWidget(lang: string, code: string): boolean {
  const l = lang.trim().toLowerCase();
  if (l === "widget") return false; // already one
  if (l !== "" && l !== "html" && l !== "htm" && l !== "xml") return false;
  return /<script[\s>]/i.test(code) || /<(button|input|select|textarea|canvas)[\s>]/i.test(code);
}

/** ```html with a script becomes ```widget (only call this when the user asked for something interactive). */
export function normalizeWidgetFences(markdown: string): string {
  return markdown.replace(
    FENCE,
    (all, indent: string, fence: string, info: string, code: string) =>
      looksLikeWidget(info, code) ? `${indent}${fence}widget\n${code}\n${indent}${fence}` : all
  );
}

/**
 * A copy of a half-written answer for the page: a widget (or html) block that has not been closed yet is replaced by a progress line, so
 * the page shows "Building the widget..." and not a growing wall of code. Once the block is closed it is shown, running, as a widget.
 */
export function previewWithProgress(markdown: string): string {
  const lines = markdown.split("\n");
  let open: { at: number; fence: string; lang: string } | null = null;
  for (let i = 0; i < lines.length; i++) {
    const m = /^[ \t]{0,3}(`{3,}|~{3,})(.*)$/.exec(lines[i]!);
    if (!m) continue;
    if (open === null) open = { at: i, fence: m[1]!, lang: m[2]!.trim().toLowerCase() };
    else if (m[1]![0] === open.fence[0] && m[1]!.length >= open.fence.length && m[2]!.trim() === "")
      open = null;
  }
  if (!open) return markdown;
  const buildable = open.lang === "widget" || open.lang === "html" || open.lang === "htm";
  if (!buildable) return markdown;
  const written = lines.length - open.at - 1;
  const before = lines.slice(0, open.at).join("\n").replace(/\s+$/, "");
  const progress = `*Building the widget... ${written > 0 ? `${written} line${written === 1 ? "" : "s"} written` : "starting"}*`;
  return before ? `${before}\n\n${progress}` : progress;
}
