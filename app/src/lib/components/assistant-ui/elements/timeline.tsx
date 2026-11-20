import type { ComponentProps } from "react";
import { take } from "../utils/range.ts";

// The Timeline element from assistant-ui (elements/timeline): events on a time axis, with what already happened and what is still
// coming. Same props and behaviour as the original; the classes use Dara's design tokens instead of shadcn's, and the row animation is
// the `timeline-row` rule in app.css.

export type TimelineWhen = "past" | "now" | "future";

export interface TimelineEvent {
  id: string;
  when: TimelineWhen;
  time: string;
  title: string;
  detail?: string | undefined;
}

const cx = (...parts: Array<string | false | undefined>) => parts.filter(Boolean).join(" ");

export function Timeline({
  events,
  visibleCount,
  className,
  ...props
}: Omit<ComponentProps<"div">, "children" | "events" | "visibleCount"> & {
  events: readonly TimelineEvent[];
  visibleCount: number;
}) {
  const shown = take(events, visibleCount);
  return (
    <div
      data-slot="timeline"
      className={cx("flex w-full flex-col rounded-lg border border-border bg-bg p-3", className)}
      {...props}
    >
      {shown.map((event, i) => (
        <div
          key={event.id}
          className="timeline-row grid grid-cols-[2.75rem_1rem_minmax(0,1fr)] gap-x-2"
        >
          <span className="pt-[3px] text-end font-mono text-[11px] tracking-tight text-text-subtle tabular-nums">
            {event.time}
          </span>

          <span className="flex flex-col items-center">
            <span
              className={cx(
                "mt-1 size-2 shrink-0 rounded-full",
                event.when === "now" &&
                  "bg-primary ring-4 ring-primary/15 forced-colors:bg-[Highlight]",
                event.when === "past" && "bg-text-muted forced-colors:bg-[CanvasText]",
                event.when === "future" && "border border-border-strong bg-transparent"
              )}
            />
            {i < shown.length - 1 && (
              <span
                className={cx(
                  "w-px flex-1",
                  event.when === "future" ? "bg-border" : "bg-border-strong"
                )}
              />
            )}
          </span>

          <div className={cx("flex flex-col gap-0.5", i < shown.length - 1 && "pb-3")}>
            <span
              className={cx(
                "break-words text-[13px]",
                event.when === "future" ? "text-text-subtle" : "text-text",
                event.when === "now" && "font-medium"
              )}
            >
              {event.title}
            </span>
            {event.detail && (
              <span className="break-words text-xs leading-relaxed text-text-muted">
                {event.detail}
              </span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
