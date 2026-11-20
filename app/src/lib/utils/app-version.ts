// The desktop shell's own version (tauri.conf.json), only known inside the Tauri window.

export function isTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

let cached: string | undefined;

/** The app version, or undefined in a browser (dev or mock mode): nothing is invented. */
export async function getAppVersion(): Promise<string | undefined> {
  if (cached !== undefined) return cached;
  if (!isTauri()) return undefined;
  try {
    const { getVersion } = await import("@tauri-apps/api/app");
    cached = await getVersion();
    return cached;
  } catch {
    return undefined;
  }
}
