import React, { useEffect, useRef, useState } from "react";
import { Icon } from "../../ui/Icon.tsx";
import { Markdown } from "../../ui/Markdown.tsx";
import { formatThinkingTime } from "../../chat/effort.ts";
import { thinkingTitle } from "../../chat/thinking.ts";

interface ReasoningProps {
  /** The model's thinking so far (markdown). */
  text: string;
  /** true while the thought is still being written. */
  streaming: boolean;
  /** How long the model thought, in ms, once it has finished. */
  durationMs?: number | undefined;
  /** The thinking budget ran out and the engine closed the thought. */
  cut?: boolean | undefined;
}

/**
 * The model's thinking, laid out like Claude's: a title taken from the thought's own latest step (it shimmers while the model is
 * thinking), with the thought indented beside a thin line. It is open while the model thinks (following the newest text unless you
 * scroll up) and folds to "Thought for 12s" when the answer starts, unless you opened or closed it yourself.
 *
 * The title row is 28px tall, the same as the Dara logo next to it, so its text sits on the logo's centre line.
 */
export const Reasoning: React.FC<ReasoningProps> = ({ text, streaming, durationMs, cut }) => {
  const [manual, setManual] = useState<boolean | null>(null); // the reader's own choice wins once made
  const open = manual ?? streaming;
  const scroller = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);

  useEffect(() => {
    const el = scroller.current;
    if (!el || !open || !streaming || !pinned.current) return;
    el.scrollTop = el.scrollHeight;
  }, [text, open, streaming]);

  const onScroll = () => {
    const el = scroller.current;
    if (!el) return;
    pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
  };

  const title = streaming
    ? thinkingTitle(text)
    : durationMs && durationMs > 0
      ? `Thought for ${formatThinkingTime(durationMs)}`
      : "Thought process";

  return (
    <div
      data-slot="reasoning-root"
      data-state={open ? "open" : "closed"}
      className="mb-2 text-left"
    >
      <button
        type="button"
        data-slot="reasoning-trigger"
        aria-expanded={open}
        onClick={() => setManual(!open)}
        className="flex h-7 max-w-full cursor-pointer items-center gap-2 rounded-md text-left text-[13px] text-text-muted transition-colors hover:text-text focus-visible:outline-2 focus-visible:outline-focus"
      >
        <span className={`truncate ${streaming ? "reasoning-shimmer" : ""}`}>{title}</span>
        {cut && !streaming && (
          <span
            className="shrink-0 rounded-full border border-border px-1.5 py-px text-[10px] font-medium text-text-subtle"
            title="The thinking limit for this effort was reached, so the model was asked to answer."
          >
            limit reached
          </span>
        )}
        <Icon
          name="chevron-down"
          size={14}
          className={`shrink-0 transition-transform duration-200 ${open ? "rotate-180" : ""}`}
        />
      </button>

      <div
        data-slot="reasoning-content"
        aria-busy={streaming}
        className={`grid transition-[grid-template-rows] duration-200 ease-out ${open ? "grid-rows-[1fr]" : "grid-rows-[0fr]"}`}
      >
        <div className="min-h-0 overflow-hidden">
          <div className="relative mt-1 border-l border-border-strong/60 pl-4">
            <div
              ref={scroller}
              onScroll={onScroll}
              data-slot="reasoning-text"
              className="reasoning-body max-h-72 overflow-y-auto pr-1 text-xs leading-relaxed text-text-muted custom-scrollbar"
            >
              <Markdown content={text} />
            </div>
            {streaming && (
              <div
                data-slot="reasoning-fade"
                aria-hidden="true"
                className="pointer-events-none absolute inset-x-0 bottom-0 h-6 bg-linear-to-t from-bg to-transparent"
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
