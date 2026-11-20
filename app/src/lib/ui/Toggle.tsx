import React from "react";

export interface ToggleProps {
  label: string;
  checked: boolean;
  onChange?: (checked: boolean) => void;
  onchange?: (checked: boolean) => void;
  description?: string;
  disabled?: boolean;
  className?: string;
}

export const Toggle: React.FC<ToggleProps> = ({
  label,
  checked,
  onChange,
  onchange,
  description,
  disabled = false,
  className = ""
}) => {
  const handler = onChange || onchange;

  return (
    <label
      className={`flex items-start justify-between gap-3 cursor-pointer select-none ${
        disabled ? "opacity-50 cursor-not-allowed" : ""
      } ${className}`}
    >
      <div className="space-y-0.5 text-left">
        <span className="text-xs font-semibold text-text block">{label}</span>
        {description && (
          <span className="text-[11px] text-text-muted block leading-tight">{description}</span>
        )}
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => !disabled && handler?.(!checked)}
        className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
          checked ? "bg-text" : "bg-surface-sunken border border-border"
        }`}
      >
        <span
          className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow-sm ring-0 transition duration-200 ease-in-out ${
            checked ? "translate-x-4" : "translate-x-0"
          }`}
        />
      </button>
    </label>
  );
};
