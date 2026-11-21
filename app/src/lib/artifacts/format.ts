import type { ArtifactKind } from "../api/types.ts";

export const KIND_LABEL: Record<ArtifactKind, string> = {
  doc: "Docs",
  slides: "Slides",
  design: "Design"
};

/** "now", "5 minutes ago", "3 hours ago", "Yesterday", "Nov 17", "Nov 17, 2025". */
export function relativeTime(ms: number, now: number = Date.now()): string {
  const diff = Math.max(0, now - ms);
  const min = Math.floor(diff / 60000);
  if (min < 1) return "now";
  if (min < 60) return `${min} minute${min === 1 ? "" : "s"} ago`;
  const hours = Math.floor(min / 60);
  if (hours < 24) return `${hours} hour${hours === 1 ? "" : "s"} ago`;
  const d = new Date(ms);
  const n = new Date(now);
  const yesterday = new Date(n.getFullYear(), n.getMonth(), n.getDate() - 1);
  if (
    d.getFullYear() === yesterday.getFullYear() &&
    d.getMonth() === yesterday.getMonth() &&
    d.getDate() === yesterday.getDate()
  )
    return "Yesterday";
  const label = d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  return d.getFullYear() === n.getFullYear() ? label : `${label}, ${d.getFullYear()}`;
}

/** A file name for an exported document: the title with characters Windows forbids removed. */
export function exportFileName(title: string, ext = "md"): string {
  const base = title
    .replace(/[<>:"/\\|?*\u0000-\u001f]/g, "")
    .trim()
    .replace(/\s+/g, " ")
    .slice(0, 80);
  return `${base || "Untitled"}.${ext}`;
}

/** The first words of a document as a plain title suggestion for the gallery when it has no title. */
export function displayTitle(title: string): string {
  return title.trim() || "Untitled";
}

/** "21 Nov, 13:16": when a comment was written. */
export function commentTime(ms: number): string {
  const d = new Date(ms);
  const day = d.toLocaleDateString("en-GB", { day: "numeric", month: "short" });
  const time = d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", hour12: false });
  return `${day}, ${time}`;
}
