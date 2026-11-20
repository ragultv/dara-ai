// Display formatting only. Nothing here derives a server fact: it just renders numbers the API returned.

/** Megabytes to "940 MB" or "2.9 GB" (binary, as the server reports sizes). */
export function formatMb(mb: number): string {
  if (!Number.isFinite(mb)) return "unknown";
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  return `${Math.round(mb)} MB`;
}

/** Gigabytes with one decimal ("1.1 GB"). */
export function formatGb(gb: number): string {
  if (!Number.isFinite(gb)) return "unknown";
  return `${gb.toFixed(1)} GB`;
}

/** Context size in thousands ("4K", "128K"); falls back to the exact number when not a multiple of 1024. */
export function formatContext(tokens: number): string {
  if (!Number.isFinite(tokens)) return "unknown";
  if (tokens >= 1024 && tokens % 1024 === 0) return `${tokens / 1024}K`;
  return tokens.toLocaleString("en-US");
}

/** Exact token count for hover titles ("131,072 tokens"). */
export function formatTokensExact(tokens: number): string {
  return `${tokens.toLocaleString("en-US")} tokens`;
}

/** Tokens per second with one decimal. */
export function formatTps(tps: number): string {
  if (!Number.isFinite(tps)) return "unknown";
  return `${tps.toFixed(1)} tok/s`;
}

/** Download rate in MB per second with one decimal. */
export function formatRate(mbPerSecond: number): string {
  if (!Number.isFinite(mbPerSecond)) return "unknown";
  return `${mbPerSecond.toFixed(1)} MB/s`;
}

/** Remaining time such as "45 s", "3 min" or "1 h 5 min". */
export function formatEta(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "unknown";
  const s = Math.round(seconds);
  if (s < 60) return `${s} s`;
  const minutes = Math.round(s / 60);
  if (minutes < 60) return `${minutes} min`;
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return rest === 0 ? `${hours} h` : `${hours} h ${rest} min`;
}

/** Uptime such as "42 s", "5 min 3 s" or "2 h 10 min". */
export function formatUptime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "unknown";
  const s = Math.floor(seconds);
  if (s < 60) return `${s} s`;
  const minutes = Math.floor(s / 60);
  if (minutes < 60) return `${minutes} min ${s % 60} s`;
  const hours = Math.floor(minutes / 60);
  if (hours < 48) return `${hours} h ${minutes % 60} min`;
  return `${Math.floor(hours / 24)} d ${hours % 24} h`;
}

/** Whole number with thousands separators ("12,345"). */
export function formatInt(n: number): string {
  if (!Number.isFinite(n)) return "unknown";
  return Math.round(n).toLocaleString("en-US");
}

/** Milliseconds as "850 ms" or "1.25 s". */
export function formatMs(ms: number): string {
  if (!Number.isFinite(ms)) return "unknown";
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
  return `${Math.round(ms)} ms`;
}

/** Percentage with no decimals ("94%"). */
export function formatPercent(value: number): string {
  if (!Number.isFinite(value)) return "unknown";
  return `${Math.round(value)}%`;
}

/** A byte count as "4.0 KB", "12.3 MB" or "1.2 GB". */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "unknown";
  if (bytes < 1024) return `${Math.round(bytes)} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}
