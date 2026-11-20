import React from "react";
import { Icon } from "./Icon.tsx";

export interface SelectOption {
  value: string;
  label: string;
  disabled?: boolean;
}

export interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  options: SelectOption[];
  hint?: string;
  error?: string;
}

export const Select = React.forwardRef<HTMLSelectElement, SelectProps>(
  ({ id, label, options, hint, error, disabled = false, className = "", ...rest }, ref) => {
    const selectId = id || `select-${Math.random().toString(36).substring(2, 9)}`;

    return (
      <div className="flex flex-col space-y-1.5 w-full text-left">
        {label && (
          <label htmlFor={selectId} className="text-xs font-semibold text-text">
            {label}
          </label>
        )}

        <div className="relative">
          <select
            ref={ref}
            id={selectId}
            disabled={disabled}
            className={`w-full h-10 pl-3 pr-8 py-2 text-sm rounded-lg bg-surface border transition-colors text-text appearance-none disabled:opacity-50 disabled:cursor-not-allowed focus-visible:outline-2 focus-visible:outline-focus focus-visible:outline-offset-2 ${
              error ? "border-danger" : "border-border focus-visible:border-primary"
            } ${className}`}
            {...rest}
          >
            {options.map((opt) => (
              <option key={opt.value} value={opt.value} disabled={opt.disabled}>
                {opt.label}
              </option>
            ))}
          </select>
          <div className="absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none text-text-subtle">
            <Icon name="chevron-down" size={13} />
          </div>
        </div>

        {error ? (
          <span className="text-xs text-danger font-medium">{error}</span>
        ) : hint ? (
          <span className="text-[11px] text-text-subtle">{hint}</span>
        ) : null}
      </div>
    );
  }
);

Select.displayName = "Select";
