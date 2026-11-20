import { useCallback, useEffect, useRef, useState } from "react";
import { apiClient } from "../../api/client.ts";
import type { BenchmarkJob } from "../../api/types.ts";
import { isBenchmarkFinished } from "./optionsLogic.ts";

const POLL_MS = 1500;
const MAX_POLL_FAILURES = 5;

export interface BenchmarkState {
  job: BenchmarkJob | null;
  isRunning: boolean;
  error: string | null;
  start: (opts: { vision: boolean; force: boolean }) => Promise<void>;
  cancel: () => Promise<void>;
}

/** Starts a benchmark and polls it; the timer is cleared when it ends and on unmount. */
export function useBenchmark(
  modelId: string,
  onFinished: (job: BenchmarkJob) => void
): BenchmarkState {
  const [job, setJob] = useState<BenchmarkJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isMounted = useRef(true);
  const finished = useRef(onFinished);

  useEffect(() => {
    finished.current = onFinished;
  }, [onFinished]);

  const stopTimer = useCallback((): void => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = null;
  }, []);

  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
      stopTimer();
    };
  }, [stopTimer]);

  const poll = useCallback(
    (jobId: string, failures: number): void => {
      timer.current = setTimeout(() => {
        void (async () => {
          try {
            const next = await apiClient.getBenchmarkJob(jobId);
            if (!isMounted.current) return;
            setJob(next);
            if (isBenchmarkFinished(next.status)) {
              stopTimer();
              finished.current(next);
              return;
            }
            poll(jobId, 0);
          } catch (err: unknown) {
            if (!isMounted.current) return;
            if (failures + 1 >= MAX_POLL_FAILURES) {
              setError(
                err instanceof Error ? err.message : "Lost contact with the engine while measuring."
              );
              setJob(null);
              return;
            }
            poll(jobId, failures + 1);
          }
        })();
      }, POLL_MS);
    },
    [stopTimer]
  );

  const start = useCallback(
    async (opts: { vision: boolean; force: boolean }): Promise<void> => {
      setError(null);
      stopTimer();
      try {
        const created = await apiClient.startBenchmark(modelId, opts);
        if (!isMounted.current) return;
        setJob(created);
        if (isBenchmarkFinished(created.status)) finished.current(created);
        else poll(created.id, 0);
      } catch (err: unknown) {
        if (isMounted.current)
          setError(err instanceof Error ? err.message : "Could not start the measurement.");
      }
    },
    [modelId, poll, stopTimer]
  );

  const cancel = useCallback(async (): Promise<void> => {
    if (!job) return;
    try {
      await apiClient.cancelBenchmark(job.id);
    } catch (err: unknown) {
      if (isMounted.current)
        setError(err instanceof Error ? err.message : "Could not cancel the measurement.");
    }
    // The next poll reports the final "cancelled" status and runs onFinished.
  }, [job]);

  return { job, isRunning: job !== null && !isBenchmarkFinished(job.status), error, start, cancel };
}
