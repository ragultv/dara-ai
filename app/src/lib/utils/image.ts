// Image helpers for attachments. The model resizes images itself (a few hundred tokens each), so large photos are shrunk here first:
// less data on every request and in the database, same answers.

export const ACCEPTED_IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp"] as const;
export const MAX_IMAGE_BYTES = 20 * 1024 * 1024;
/** Photos above this size are downscaled before they are attached. */
const DOWNSCALE_ABOVE_BYTES = 1_500_000;
const MAX_EDGE = 1600;

export function isAcceptedImage(file: Pick<File, "type">): boolean {
  return (ACCEPTED_IMAGE_TYPES as readonly string[]).includes(file.type);
}

function readAsDataUri(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Could not read the image file"));
    reader.onload = () => {
      if (typeof reader.result === "string") resolve(reader.result);
      else reject(new Error("Could not read the image file"));
    };
    reader.readAsDataURL(blob);
  });
}

async function downscale(file: File): Promise<string> {
  const bitmap = await createImageBitmap(file);
  try {
    const scale = Math.min(1, MAX_EDGE / Math.max(bitmap.width, bitmap.height));
    const w = Math.max(1, Math.round(bitmap.width * scale));
    const h = Math.max(1, Math.round(bitmap.height * scale));
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return readAsDataUri(file);
    ctx.fillStyle = "#ffffff"; // transparent PNGs become white instead of black when saved as JPEG
    ctx.fillRect(0, 0, w, h);
    ctx.drawImage(bitmap, 0, 0, w, h);
    return canvas.toDataURL("image/jpeg", 0.88);
  } finally {
    bitmap.close();
  }
}

/** A `data:` URI ready to display and send. Throws a user-readable Error for unsupported or oversized files. */
export async function prepareImage(file: File): Promise<string> {
  if (!isAcceptedImage(file))
    throw new Error(`${file.name || "This file"} is not a PNG, JPEG or WebP image.`);
  if (file.size > MAX_IMAGE_BYTES)
    throw new Error(`${file.name || "This image"} is larger than 20 MB.`);
  if (file.size > DOWNSCALE_ABOVE_BYTES && typeof createImageBitmap === "function") {
    try {
      return await downscale(file);
    } catch {
      // fall through to the original bytes
    }
  }
  return readAsDataUri(file);
}

const dataUriCache = new Map<string, string>();

/** `data:` URIs pass through; blob: URLs (images loaded from the database) are converted so they can go in a request. */
export async function toDataUri(url: string): Promise<string> {
  if (url.startsWith("data:")) return url;
  const hit = dataUriCache.get(url);
  if (hit) return hit;
  const res = await fetch(url);
  const uri = await readAsDataUri(await res.blob());
  if (dataUriCache.size > 24) dataUriCache.clear();
  dataUriCache.set(url, uri);
  return uri;
}
