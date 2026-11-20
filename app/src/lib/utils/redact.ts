// Privacy helpers for text the user copies out of the app (Monitor diagnostics, error details).
// Rule (docs/ui/05-security.md): paths are reduced to file names; no prompts, no user names, no keys.

/**
 * Redacts the user-name segment of profile paths in free text:
 * C:\Users\<user>\... , /home/<user>/... , /Users/<user>/...
 */
export function redactUserPaths(text: string): string {
  return (
    text
      // Escaped JSON Windows paths: C:\\Users\\<user>\\
      .replace(/([a-zA-Z]:\\\\(?:Users|users)\\\\)([^\\\\"\s]+)/gi, "$1<redacted>")
      // Normal Windows paths: C:\Users\<user>\
      .replace(/([a-zA-Z]:\\(?:Users|users)\\)([^\\"\s]+)/gi, "$1<redacted>")
      // Unix user directories: /home/<user>/ or /Users/<user>/
      .replace(/((\/|^)(?:home|Users)\/)([^/"\s]+)/g, "$1<redacted>")
  );
}

const WINDOWS_PATH = /^[a-zA-Z]:[\\/]/;
const UNC_PATH = /^\\\\/;
const UNIX_USER_PATH = /^\/(?:home|Users)\//;
// A path token inside a longer sentence ("failed to open D:\a\b\model.gguf").
const EMBEDDED_WINDOWS_PATH = /[a-zA-Z]:[\\/][^\s"'<>|*?]*/g;
const EMBEDDED_UNIX_USER_PATH = /\/(?:home|Users)\/[^\s"'<>|*?]*/g;

/** True when the whole string is an absolute file path. */
export function isAbsolutePath(value: string): boolean {
  return WINDOWS_PATH.test(value) || UNC_PATH.test(value) || UNIX_USER_PATH.test(value);
}

/** "C:\Users\someone\AppData\models\nano.gguf" becomes "nano.gguf". Handles both slash styles. */
export function pathToFileName(path: string): string {
  const parts = path.split(/[\\/]+/).filter((p) => p.length > 0);
  return parts[parts.length - 1] ?? "";
}

/** Reduces every path inside a sentence to its file name, then masks any leftover user folder. */
export function scrubText(text: string): string {
  const reduced = text
    .replace(EMBEDDED_WINDOWS_PATH, (m) => pathToFileName(m))
    .replace(EMBEDDED_UNIX_USER_PATH, (m) => pathToFileName(m));
  return redactUserPaths(reduced);
}

const SECRET_KEY = /api[_-]?key|authorization|password|secret|^token$|_token$/i;

function sanitizeString(value: string): string {
  return isAbsolutePath(value) ? pathToFileName(value) : scrubText(value);
}

/** Deep copy of any JSON value where every string is path-scrubbed. Keys named like secrets are dropped. */
export function sanitizeValue(value: unknown): unknown {
  if (typeof value === "string") return sanitizeString(value);
  if (Array.isArray(value)) return value.map(sanitizeValue);
  if (value !== null && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [key, v] of Object.entries(value as Record<string, unknown>)) {
      if (SECRET_KEY.test(key)) continue;
      out[key] = sanitizeValue(v);
    }
    return out;
  }
  return value;
}

export interface DiagnosticsExtras {
  /** App (desktop shell) version. */
  appVersion?: string | undefined;
  /** Non-secret configuration worth including (defaults, engine settings). */
  config?: Record<string, unknown> | undefined;
  /** Recent error messages shown in the UI. */
  errors?: string[] | undefined;
}

/**
 * Builds the text block for "Copy diagnostics": versions, GPU name and VRAM, configuration, counters
 * and errors. `stats` and `version` are the raw API responses. Never includes prompts, the API key or
 * a user name; paths become file names.
 */
export function formatDiagnostics(
  stats: unknown,
  version: unknown,
  extras: DiagnosticsExtras = {}
): string {
  const combined: Record<string, unknown> = {
    diagnostics_created_at: new Date().toISOString(),
    app_version: extras.appVersion,
    engine_version: version,
    stats,
    config: extras.config,
    errors: extras.errors && extras.errors.length > 0 ? extras.errors : undefined
  };
  const json = JSON.stringify(sanitizeValue(combined), null, 2);
  return redactUserPaths(json);
}
