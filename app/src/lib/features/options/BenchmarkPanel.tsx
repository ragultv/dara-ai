import React, { useCallback, useState } from "react";
import type { OptionsResponse } from "../../api/types.ts";
import { Button } from "../../ui/Button.tsx";
import { Dialog } from "../../ui/Dialog.tsx";
import { Icon } from "../../ui/Icon.tsx";
import type { BenchmarkState } from "./useBenchmark.ts";

interface Props {
  modelName: string;
  calibration: OptionsResponse["calibration"] | null;
  vision: boolean;
  isBlocked: boolean;
  bench: BenchmarkState;
}

const LOG_LINES = 40;

export const BenchmarkPanel: React.FC<Props> = ({
  modelName,
  calibration,
  vision,
  isBlocked,
  bench
}) => {
  const [isConfirmOpen, setIsConfirmOpen] = useState(false);
  const { job, isRunning, error, start, cancel } = bench;
  const closeConfirm = useCallback(() => setIsConfirmOpen(false), []);
  const status = calibration?.status ?? "none";
  const hasMeasurements = status === "measured" || status === "partial";

  const confirm = (): void => {
    setIsConfirmOpen(false);
    void start({ vision, force: status === "measured" });
  };

  return (
    <section
      aria-labelledby="bench-title"
      className="rounded-2xl border border-border bg-surface p-4 space-y-3"
    >
      <h2 id="bench-title" className="text-sm font-bold text-text">
        Measure speed on this PC
      </h2>
      {calibration?.note && (
        <p className="text-xs text-text-muted leading-snug">{calibration.note}</p>
      )}
      {status === "stale" && (
        <p className="text-xs text-text leading-snug">
          The saved speeds were measured on another GPU, driver or version, so they may not match
          this PC.
        </p>
      )}

      {!isRunning && (
        <Button
          variant="secondary"
          size="md"
          disabled={isBlocked}
          onClick={() => setIsConfirmOpen(true)}
        >
          <Icon name="activity" size={14} />
          {hasMeasurements ? "Measure again on this PC" : "Measure on this PC"}
        </Button>
      )}

      {error && (
        <p role="alert" className="text-xs text-danger">
          {error}
        </p>
      )}

      {job && (
        <div className="space-y-2">
          <p role="status" className="text-xs font-semibold text-text">
            {isRunning
              ? "Measuring. The model is being unloaded and reloaded by the engine while this runs."
              : job.status === "done"
                ? "Measurement finished. The numbers above are updated."
                : job.status === "cancelled"
                  ? "Measurement cancelled."
                  : `Measurement failed${job.error ? `: ${job.error}` : "."}`}
          </p>
          {job.log.length > 0 && (
            <pre
              role="log"
              aria-label="Measurement log"
              className="max-h-48 overflow-auto rounded-lg bg-code-bg border border-border p-2 text-[11px] font-mono text-text whitespace-pre-wrap"
            >
              {job.log.slice(-LOG_LINES).join("\n")}
            </pre>
          )}
          {isRunning && (
            <Button variant="secondary" size="sm" onClick={() => void cancel()}>
              Cancel measuring
            </Button>
          )}
        </div>
      )}

      <Dialog
        isOpen={isConfirmOpen}
        title={`Measure ${modelName} on this PC?`}
        description="This takes several minutes and uses your GPU heavily."
        onclose={closeConfirm}
        actions={
          <>
            <Button variant="ghost" size="sm" onClick={closeConfirm}>
              Cancel
            </Button>
            <Button variant="primary" size="sm" onClick={confirm}>
              Start measuring
            </Button>
          </>
        }
      >
        <ul className="list-disc pl-5 text-xs space-y-1">
          <li>If this is a laptop, plug it in first. Speeds measured on battery may be lower.</li>
          <li>
            The engine unloads the current model and reloads it afterwards, so chat is unavailable
            meanwhile.
          </li>
          <li>You can cancel at any time.</li>
        </ul>
      </Dialog>
    </section>
  );
};
