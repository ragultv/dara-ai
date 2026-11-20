import React, { useState } from "react";
import { Button } from "../../ui/Button.tsx";
import { toast } from "../../ui/Toast.tsx";

interface ApiKeyFieldProps {
  /** The real key of this run, or null when the app is not managing the engine. */
  apiKey: string | null;
  /** Present only when the app can really restart the engine with a new key. */
  onRegenerate?: (() => void) | undefined;
}

const MASK = "•".repeat(32);

export const ApiKeyField: React.FC<ApiKeyFieldProps> = ({ apiKey, onRegenerate }) => {
  const [isRevealed, setIsRevealed] = useState(false);

  if (apiKey === null) {
    return (
      <p className="text-xs text-text-muted">
        No key: the app is not managing the engine in this window (browser or mock mode).
      </p>
    );
  }

  const copy = async (): Promise<void> => {
    try {
      await navigator.clipboard.writeText(apiKey);
      toast.show("API key copied. Treat it like a password.", "success");
    } catch {
      toast.show("The clipboard is not available.", "error");
    }
  };

  return (
    <div className="space-y-2 rounded-xl border border-border bg-surface-sunken p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span id="api-key-label" className="text-xs font-semibold text-text">
          Engine API key
        </span>
        <div className="flex items-center gap-1.5">
          <Button
            variant="ghost"
            size="xs"
            onClick={() => setIsRevealed((v) => !v)}
            aria-pressed={isRevealed}
          >
            {isRevealed ? "Hide" : "Reveal"}
          </Button>
          <Button variant="secondary" size="xs" onClick={() => void copy()}>
            Copy
          </Button>
          {onRegenerate && (
            <Button variant="ghost" size="xs" onClick={onRegenerate}>
              Regenerate
            </Button>
          )}
        </div>
      </div>
      <code
        aria-labelledby="api-key-label"
        className="block truncate rounded-lg border border-border bg-surface px-2.5 py-2 font-mono text-xs text-text select-all"
      >
        {isRevealed ? apiKey : MASK}
      </code>
      <p className="text-[11px] text-text-muted">
        Made fresh every time Dara starts and kept only in memory. Other programs need it to talk to
        the engine.
      </p>
    </div>
  );
};
