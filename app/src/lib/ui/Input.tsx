import React from "react";

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  hint?: string;
  error?: string;
}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ id, label, hint, error, disabled = false, className = "", type = "text", ...rest }, ref) => {
    const inputId = id || `input-${Math.random().toString(36).substring(2, 9)}`;
    const hintId = hint ? `${inputId}-hint` : undefined;
    const errorId = error ? `${inputId}-error` : undefined;
    const describedBy = [hintId, errorId].filter(Boolean).join(" ") || undefined;

    return (
      <div className="flex flex-col space-y-1.5 w-full text-left">
        {label && (
          <label htmlFor={inputId} className="text-xs font-semibold text-text">
            {label}
          </label>
        )}

        <input
          ref={ref}
          id={inputId}
          type={type}
          disabled={disabled}
          aria-invalid={Boolean(error)}
          aria-describedby={describedBy}
          className={`w-full h-10 px-3 py-2 text-sm rounded-lg bg-surface border transition-colors text-text placeholder:text-text-subtle disabled:opacity-50 disabled:cursor-not-allowed focus-visible:outline-2 focus-visible:outline-focus focus-visible:outline-offset-2 ${
            error
              ? "border-danger focus-visible:outline-danger"
              : "border-border focus-visible:border-border-strong"
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

Input.displayName = "Input";
