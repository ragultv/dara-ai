import React from "react";
import type { Stats } from "../../api/types.ts";
import { formatInt, formatTps } from "../../utils/format.ts";
import { Badge } from "../../ui/Badge.tsx";
import { KeyValue } from "./KeyValue.tsx";
import { Panel } from "./Panel.tsx";

interface EnginePanelProps {
  stats: Stats;
}

export const EnginePanel: React.FC<EnginePanelProps> = ({ stats }) => {
  const e = stats.engine;
  // With no model loaded the engine reports nothing yet.
  if (!e) {
    return (
      <Panel title="Engine" icon="activity">
        <p className="text-xs text-text-muted">
          No model is loaded, so there is nothing to report. Load one on the Models page.
        </p>
      </Panel>
    );
  }
  // An average of 0 before any request means "not measured", not "zero speed".
  const hasRequests = e.requests > 0;
  return (
    <Panel
      title="Engine"
      icon="activity"
      aside={
        <Badge variant={e.busy ? "primary" : "neutral"} size="sm">
          {e.busy ? "Generating" : "Idle"}
          {stats.queued > 0 ? ` · ${stats.queued} waiting` : ""}
        </Badge>
      }
    >
      <KeyValue
        rows={[
          { label: "Requests", value: formatInt(e.requests) },
          {
            label: "Errors",
            value: (
              <span className={e.errors > 0 ? "text-danger" : undefined}>
                {formatInt(e.errors)}
              </span>
            )
          },
          {
            label: "Context compactions",
            value: formatInt(e.compactions),
            title: "Times older messages were summarized to fit the context"
          },
          { label: "Tokens read (prompts)", value: formatInt(e.prompt_tokens_total) },
          { label: "Tokens written (replies)", value: formatInt(e.completion_tokens_total) },
          {
            label: "Tokens reused from cache",
            value: formatInt(e.cached_tokens_total),
            title: "Prompt tokens that did not need to be processed again"
          },
          {
            label: "Avg prefill speed",
            value: hasRequests ? formatTps(e.avg_prefill_tokens_per_second) : "Not measured yet"
          },
          {
            label: "Avg decode speed",
            value: hasRequests ? formatTps(e.avg_decode_tokens_per_second) : "Not measured yet"
          }
        ]}
      />
    </Panel>
  );
};
