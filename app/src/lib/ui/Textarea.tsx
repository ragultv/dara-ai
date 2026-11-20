import React from "react";

export interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;
  hint?: string;
  error?: string;
}

export const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ id, label, hint, error, disabled = false, className = "", rows = 3, ...rest }, ref) => {
    const textareaId = id || `textarea-${Math.random().toString(36).substring(2, 9)}`;
    const hintId = hint ? `${textareaId}-hint` : undefined;
    const errorId = error ? `${textareaId}-error` : undefined;
    const describedBy = [hintId, errorId].filter(Boolean).join(" ") || undefined;

    return (
      <div className="flex flex-col space-y-1.5 w-full text-left">
        {label && (
          <label htmlFor={textareaId} className="text-xs font-semibold text-text">
            {label}
          </label>
        )}

        <textarea
          ref={ref}
          id={textareaId}
          rows={rows}
          disabled={disabled}
          aria-invalid={Boolean(error)}
          aria-describedby={describedBy}
          className={`w-full p-3 text-sm rounded-lg bg-surface border transition-colors text-text placeholder:text-text-subtle disabled:opacity-50 disabled:cursor-not-allowed focus-visible:outline-2 focus-visible:outline-focus focus-visible:outline-offset-2 leading-relaxed resize-y ${
            error
              ? "border-danger focus-visible:outline-danger"
              : "border-border focus-visible:border-primary"
          } ${className}`}
          {...rest}
        />

        {error ? (
          <span id={errorId} className="text-xs text-danger font-medium">
            {error}
          </span>
        ) : hint ? (
          <span id={hintId} className="text-[11px] text-text-subtle">
            {hint}
          </span>
        ) : null}
      </div>
    );
  }
);

Textarea.displayName = "Textarea";
