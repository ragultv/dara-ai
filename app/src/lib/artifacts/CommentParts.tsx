import React, { useState } from "react";
import type { DbArtifactComment } from "../api/types.ts";
import { Icon } from "../ui/Icon.tsx";
import { commentTime } from "./format.ts";

/** The round badge before a name: Dara's is the brand colour with a spark, yours is a neutral person. */
export const Avatar: React.FC<{ author: "user" | "dara" }> = ({ author }) => (
  <span
    aria-hidden="true"
    className={`flex size-7 shrink-0 items-center justify-center rounded-full ${
      author === "dara" ? "bg-primary text-white" : "bg-surface-sunken text-text-muted"
    }`}
  >
    <Icon name={author === "dara" ? "sparkles" : "user"} size={14} />
  </span>
);

/** One comment, laid out like Claude's cards: avatar, name, date, then the words. */
export const CommentMessage: React.FC<{ c: DbArtifactComment }> = ({ c }) => (
  <div className="flex gap-2.5">
    <Avatar author={c.author} />
    <div className="min-w-0 flex-1">
      <p className="flex items-baseline gap-2 leading-7">
        <span className="text-sm font-semibold text-text">
          {c.author === "dara" ? "Dara" : "You"}
        </span>
        <span className="text-xs text-text-subtle">{commentTime(c.created_at)}</span>
      </p>
      <p className="whitespace-pre-wrap text-[13px] leading-relaxed text-text">{c.body}</p>
    </div>
  </div>
);

/** A one-line reply field that grows; Enter sends, Shift+Enter starts a new line. */
export const ReplyBox: React.FC<{
  placeholder: string;
  disabled?: boolean;
  onSubmit: (body: string) => void;
}> = ({ placeholder, disabled, onSubmit }) => {
  const [text, setText] = useState("");
  const submit = () => {
    const v = text.trim();
    if (!v || disabled) return;
    setText("");
    onSubmit(v);
  };
  return (
    <div className="flex items-end gap-1.5 rounded-lg border border-border bg-bg py-1 pl-2.5 pr-1 focus-within:border-border-strong">
      <textarea
        value={text}
        rows={1}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
        placeholder={placeholder}
        aria-label={placeholder}
        className="custom-scrollbar max-h-24 min-h-7 flex-1 resize-none border-0 bg-transparent py-1 text-[13px] text-text outline-none placeholder:text-text-subtle"
      />
      <button
        type="button"
        onClick={submit}
        disabled={!text.trim() || disabled}
        aria-label="Reply"
        title="Reply"
        className="flex size-7 shrink-0 cursor-pointer items-center justify-center rounded-md bg-primary text-white transition-colors hover:bg-primary-hover disabled:cursor-not-allowed disabled:opacity-40"
      >
        <Icon name="arrow-up" size={14} />
      </button>
    </div>
  );
};
