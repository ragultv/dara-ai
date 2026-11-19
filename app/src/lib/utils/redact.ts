/**
 * Redacts user profile paths from diagnostics strings or objects
 * to ensure privacy and prevent username leakage in crash reports / diagnostic exports.
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

export function formatDiagnostics(stats: unknown, version: unknown): string {
  const combined = {
    diagnostics_created_at: new Date().toISOString(),
    version,
    stats
  };
  const json = JSON.stringify(combined, null, 2);
  return redactUserPaths(json);
}
