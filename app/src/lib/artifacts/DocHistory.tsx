import React, { useCallback, useEffect, useState } from "react";
import { storeApi } from "../api/store-api.ts";
import type { DbArtifactVersion, DbArtifactVersionSummary } from "../api/types.ts";
import { useArtifactsStore } from "../stores/useArtifactsStore.ts";
import { Markdown } from "../ui/Markdown.tsx";
import { Button } from "../ui/Button.tsx";
import { Skeleton } from "../ui/Skeleton.tsx";
import { toast } from "../ui/Toast.tsx";
import { relativeTime } from "./format.ts";
import { collabMessage } from "./errors.ts";

interface Props {
  docId: string;
  /** changes whenever the document is saved, so the list follows the edits */
  updatedAt: number;
  /** called with the restored document so the editor can show it */
  onRestored: (markdown: string) => void;
}

/** Version history: the earlier states the server kept, a preview of one, and a button to put it back. */
export const DocHistory: React.FC<Props> = ({ docId, updatedAt, onRestored }) => {
  const restore = useArtifactsStore((s) => s.restore);
  const [items, setItems] = useState<DbArtifactVersionSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [picked, setPicked] = useState<number | null>(null);
  const [preview, setPreview] = useState<DbArtifactVersion | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      const res = await storeApi.listArtifactVersions(docId);
      setItems(res.data);
      setError(null);
    } catch (e) {
      setError(collabMessage(e, "version history"));
    }
  }, [docId]);

  useEffect(() => {
    void load();
  }, [load, updatedAt]);

  useEffect(() => {
    setPreviewError(null);
    if (picked === null) {
      setPreview(null);
      return;
    }
    let cancelled = false;
    setPreview(null);
    storeApi
      .getArtifactVersion(docId, picked)
      .then((v) => !cancelled && setPreview(v))
      .catch((e) => !cancelled && setPreviewError(collabMessage(e, "version history")));
    return () => {
      cancelled = true;
    };
  }, [docId, picked]);

  const doRestore = async () => {
    if (picked === null || busy) return;
    setBusy(true);
    const doc = await restore(picked);
    setBusy(false);
    if (doc) {
      onRestored(doc.content);
      setPicked(null);
      toast.show("Restored. The text from before is kept in the history.", "info");
      void load();
    } else {
      toast.show("Could not restore that version.", "error");
    }
  };

  if (error) {
    return (
      <div className="flex flex-col items-start gap-3 p-4 text-xs text-text-muted">
        <p role="alert" className="text-danger">
          {error}
        </p>
        <Button variant="secondary" size="xs" onClick={() => void load()}>
          Try again
        </Button>
      </div>
    );
  }
  if (items === null) {
    return (
      <div className="space-y-2 p-4">
        <Skeleton className="h-10 w-full" />
        <Skeleton className="h-10 w-full" />
        <Skeleton className="h-10 w-full" />
      </div>
    );
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <ul
        aria-label="Versions"
        className={`min-h-0 space-y-1 overflow-y-auto p-3 custom-scrollbar ${
          picked === null ? "flex-1" : "max-h-[42%] shrink-0"
        }`}
      >
        <li>
          <button
            type="button"
            aria-pressed={picked === null}
            onClick={() => setPicked(null)}
            className={`w-full cursor-pointer rounded-lg border px-3 py-2 text-left transition-colors ${
              picked === null
                ? "border-border-strong bg-primary-subtle"
                : "border-transparent hover:bg-surface-raised"
            }`}
          >
            <p className="text-xs font-semibold text-text">Current version</p>
            <p className="text-[11px] text-text-muted">Saved {relativeTime(updatedAt)}</p>
          </button>
        </li>
        {items.length === 0 && (
          <li className="px-1 pt-3 text-xs leading-relaxed text-text-muted">
            No earlier versions yet. Dara keeps one when you start editing after a pause, before it
            changes the document, and before a restore.
          </li>
        )}
        {items.map((v) => (
          <li key={v.rev}>
            <button
              type="button"
              aria-pressed={picked === v.rev}
              onClick={() => setPicked(picked === v.rev ? null : v.rev)}
              className={`w-full cursor-pointer rounded-lg border px-3 py-2 text-left transition-colors ${
                picked === v.rev
                  ? "border-border-strong bg-primary-subtle"
                  : "border-transparent hover:bg-surface-raised"
              }`}
            >
              <p className="truncate text-xs font-semibold text-text">
                {v.label || "Autosaved version"}
              </p>
              <p className="text-[11px] text-text-muted">
                {relativeTime(v.created_at)} · {v.chars.toLocaleString()} characters
              </p>
            </button>
          </li>
        ))}
      </ul>

      {picked !== null && (
        <div className="flex min-h-0 flex-1 flex-col border-t border-border">
          <p className="shrink-0 border-b border-border bg-surface-raised px-4 py-2 text-[11px] text-text-muted">
            Previewing an earlier version. Your document has not changed.
          </p>
          <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3 text-[13px] custom-scrollbar">
            {previewError ? (
              <p role="alert" className="text-xs text-danger">
                {previewError}
              </p>
            ) : preview ? (
              <>
                {preview.title && (
                  <p className="mb-2 font-[family-name:var(--font-reply)] text-base font-semibold text-text">
                    {preview.title}
                  </p>
                )}
                {preview.content.trim() ? (
                  <Markdown content={preview.content} />
                ) : (
                  <p className="text-xs text-text-muted">This version is empty.</p>
                )}
              </>
            ) : (
              <Skeleton className="h-24 w-full" />
            )}
          </div>
          <div className="flex shrink-0 items-center justify-end gap-2 border-t border-border p-3">
            <Button variant="ghost" size="sm" onClick={() => setPicked(null)}>
              Back to current
            </Button>
            <Button
              variant="primary"
              size="sm"
              disabled={busy || !preview}
              onClick={() => void doRestore()}
            >
              Restore this version
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};
