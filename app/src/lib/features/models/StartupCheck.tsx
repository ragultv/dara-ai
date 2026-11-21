import React, { useEffect, useRef, useState } from "react";
import { apiClient } from "../../api/client.ts";
import { useServerStore } from "../../stores/useServerStore.ts";
import { Button } from "../../ui/Button.tsx";
import { Icon } from "../../ui/Icon.tsx";
import { startupProblem, type StartupProblem } from "./startupCheck.ts";

/**
 * First-run check at the top of onboarding: says plainly when the engine did not start, there is no NVIDIA graphics card, or the
 * card is too old, instead of leaving the user with models that will not load. Nothing is shown when all is well.
 */
export const StartupCheck: React.FC = () => {
  const status = useServerStore((s) => s.status);
  const init = useServerStore((s) => s.init);
  const [problem, setProblem] = useState<StartupProblem | null>(null);
  const started = useRef(Date.now());

  useEffect(() => {
    let cancelled = false;
    const run = async (): Promise<void> => {
      const engineUp = useServerStore.getState().status === "ready";
      let stats = null;
      if (engineUp) {
        try {
          stats = await apiClient.getStats();
        } catch {
          stats = null;
        }
      }
      if (cancelled) return;
      setProblem(startupProblem({ engineUp, stats, waitedMs: Date.now() - started.current }));
    };
    void run();
    // an engine that is still starting is checked again until it answers or time runs out
    const timer = window.setInterval(() => void run(), 3000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [status]);

  if (!problem) return null;

  const openHelp = (): void => {
    if (!problem.helpUrl) return;
    void import("@tauri-apps/plugin-opener")
      .then((m) => m.openUrl(problem.helpUrl as string))
      .catch(() => window.open(problem.helpUrl, "_blank", "noopener"));
  };

  return (
    <div
      role="alert"
      className="w-full max-w-xl space-y-2 rounded-xl border border-border bg-warning-subtle p-4 text-left"
    >
      <div className="flex items-center gap-2 text-sm font-semibold text-text">
        <Icon name="alert-triangle" size={16} className="text-warning" />
        <span>{problem.title}</span>
      </div>
      <p className="text-xs leading-relaxed text-text-muted">{problem.body}</p>
      <div className="flex flex-wrap gap-2 pt-1">
        {problem.helpUrl && (
          <Button size="sm" variant="secondary" onClick={openHelp}>
            Get the latest driver
          </Button>
        )}
        <Button
          size="sm"
          variant="ghost"
          onClick={() => {
            started.current = Date.now();
            void init();
          }}
        >
          Check again
        </Button>
      </div>
    </div>
  );
};
