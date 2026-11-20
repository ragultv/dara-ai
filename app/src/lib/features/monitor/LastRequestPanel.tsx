import React from "react";
import type { ContextInfo, Stats, Timings, Usage } from "../../api/types.ts";
import { formatContext, formatInt, formatMs, formatTps } from "../../utils/format.ts";
import { Badge } from "../../ui/Badge.tsx";
import { KeyValue } from "./KeyValue.tsx";
import { Panel } from "./Panel.tsx";

interface LastRequestPanelProps {
  last: Stats["last_request"];
}

export const LastRequestPanel: React.FC<LastRequestPanelProps> = ({ last }) => {
  if (!last) {
    return (
      <Panel title="Last request" icon="message-square">
        <p className="text-xs text-text-muted">
          No request has been served since the engine started.
        </p>
      </Panel>
    );
  }
  // The API tolerates missing groups, so each one is checked before it is shown.
  const t = last.timings as Timings | undefined;
  const u = last.usage as Usage | undefined;
  const c = last.context as ContextInfo | undefined;
  const s = last.speculative;
  return (
    <Panel
      title="Last request"
      icon="message-square"
      aside={
        s ? (
          <Badge variant={s.flash_stood_down_at_pass !== null ? "warning" : "primary"} size="sm">
            {s.flash_stood_down_at_pass !== null ? "Flash stood down" : `Speculative: ${s.mode}`}
          </Badge>
        ) : undefined
      }
    >
      <div className="grid gap-x-8 gap-y-4 md:grid-cols-2">
        {t && (
          <KeyValue
            rows={[
              { label: "Waited in queue", value: formatMs(t.queue_ms) },
              { label: "Time to first token", value: formatMs(t.ttft_ms) },
              {
                label: "Prefill",
                value: `${formatInt(t.prefill_tokens)} tokens · ${formatTps(t.prefill_tokens_per_second)}`
              },
              {
                label: "Decode",
                value: `${formatInt(t.decode_tokens)} tokens · ${formatTps(t.decode_tokens_per_second)}`
              },
              { label: "Total time", value: formatMs(t.total_ms) }
            ]}
          />
        )}
        <KeyValue
          rows={[
            { label: "Prompt tokens", value: u ? formatInt(u.prompt_tokens) : undefined },
            { label: "Reply tokens", value: u ? formatInt(u.completion_tokens) : undefined },
            {
              label: "Reused from cache",
              value: u?.prompt_tokens_details
                ? formatInt(u.prompt_tokens_details.cached_tokens)
                : undefined
            },
            {
              label: "Context used",
              value: c ? `${formatInt(c.used)} of ${formatContext(c.capacity)}` : undefined
            },
            { label: "Compacted", value: c ? (c.compacted ? "Yes" : "No") : undefined }
          ]}
        />
      </div>
      {s && (
        <div className="border-t border-border pt-3">
          <KeyValue
            rows={[
              { label: "Draft acceptance", value: `${s.acceptance.toFixed(1)}%` },
              { label: "Tokens per pass", value: s.tokens_per_pass.toFixed(2) },
              {
                label: "Drafted / accepted",
                value: `${formatInt(s.drafted)} / ${formatInt(s.accepted)}`
              },
              {
                label: "Flash stood down",
                value:
                  s.flash_stood_down_at_pass !== null
                    ? `Yes, at pass ${s.flash_stood_down_at_pass}`
                    : "No",
                title: "Flash pauses itself when it is not paying off; this is not an error"
              }
            ]}
          />
          {s.note && <p className="mt-2 text-[11px] text-text-muted">{s.note}</p>}
        </div>
      )}
    </Panel>
  );
};
