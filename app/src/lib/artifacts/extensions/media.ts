import Image from "@tiptap/extension-image";
import type { Editor } from "@tiptap/react";
import { storeApi } from "../../api/store-api.ts";
import { collabMessage } from "../errors.ts";

// Images and video in a document. The file is stored with the document on this computer and the Markdown only says where it is:
//   ![name](dara-asset:<id>)            an image
//   ![name](dara-asset:<id> "video")    a video
// A link to anywhere else on the internet is never loaded (nothing leaves this computer, and a document written by the model cannot
// make the app contact a server).

const ASSET_RE = /^dara-asset:([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$/i;

export const MAX_MEDIA_BYTES = 25 * 1024 * 1024;
export const MEDIA_TYPES = [
  "image/png",
  "image/jpeg",
  "image/gif",
  "image/webp",
  "video/mp4",
  "video/webm"
] as const;
export const MEDIA_ACCEPT = MEDIA_TYPES.join(",");

export function isMediaFile(file: { type: string }): boolean {
  return (MEDIA_TYPES as readonly string[]).includes(file.type);
}

export function assetIdOf(src: string): string | null {
  return ASSET_RE.exec(src)?.[1]?.toLowerCase() ?? null;
}

const urls = new Map<string, Promise<string>>();

/** A blob URL for a stored asset (the request carries the API key, which an <img src> cannot). Cached for the session. */
export function assetUrl(docId: string, assetId: string): Promise<string> {
  const key = `${docId}/${assetId}`;
  let p = urls.get(key);
  if (!p) {
    p = storeApi.getArtifactAsset(docId, assetId).then((blob) => URL.createObjectURL(blob));
    p.catch(() => urls.delete(key));
    urls.set(key, p);
  }
  return p;
}

export const DaraMedia = Image.extend<{
  getDocId: () => string;
  inline: boolean;
  allowBase64: boolean;
  HTMLAttributes: Record<string, unknown>;
}>({
  addOptions() {
    return { ...this.parent?.(), getDocId: () => "" } as never;
  },
  addNodeView() {
    const getDocId = this.options.getDocId;
    return ({ node }) => {
      const src = String(node.attrs.src ?? "");
      const alt = String(node.attrs.alt ?? "");
      const isVideo = node.attrs.title === "video";
      const dom = document.createElement("figure");
      dom.className = "doc-media";
      const note = (text: string) => {
        dom.replaceChildren();
        const p = document.createElement("p");
        p.className = "doc-media-note";
        p.textContent = text;
        dom.append(p);
      };
      const id = assetIdOf(src);
      if (!id) {
        note(`Not shown: ${alt || "media"} is not stored in this document.`);
        return { dom };
      }
      const el = document.createElement(isVideo ? "video" : "img");
      if (el instanceof HTMLVideoElement) {
        el.controls = true;
        el.preload = "metadata";
      } else {
        el.alt = alt;
        el.draggable = false;
      }
      dom.append(el);
      assetUrl(getDocId(), id)
        .then((u) => {
          el.src = u;
        })
        .catch(() => note(`Could not load ${alt || "this file"}.`));
      return { dom };
    };
  }
});

/** Stores a picked, pasted or dropped file with the document and puts it into the page. Returns a message when it cannot. */
export async function insertMedia(
  editor: Editor,
  docId: string,
  file: File
): Promise<string | null> {
  if (!isMediaFile(file))
    return "Only PNG, JPEG, GIF, WebP images and MP4 or WebM video can be added.";
  if (file.size > MAX_MEDIA_BYTES) return "That file is over 25 MB.";
  try {
    const asset = await storeApi.uploadArtifactAsset(docId, file, file.name || "file");
    const isVideo = asset.mime.startsWith("video/");
    // after the block the cursor is in: an image cannot go inside a heading, a list item or a table cell
    const { $to } = editor.state.selection;
    const pos = $to.depth >= 1 ? $to.after(1) : $to.pos;
    editor
      .chain()
      .focus()
      .insertContentAt(pos, {
        type: "image",
        attrs: { src: `dara-asset:${asset.id}`, alt: asset.name, title: isVideo ? "video" : null }
      })
      .run();
    return null;
  } catch (e) {
    return collabMessage(e, "images and video");
  }
}
