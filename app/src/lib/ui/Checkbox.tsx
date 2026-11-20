import React from "react";
import { Icon } from "./Icon.tsx";

export interface CheckboxProps {
  id?: string;
  label?: string;
  description?: string;
  checked?: boolean;
  disabled?: boolean;
  className?: string;
  onchange?: (checked: boolean) => void;
}

export const Checkbox: React.FC<CheckboxProps> = ({
  id,
  label,
  description,
  checked = false,
  disabled = false,
  className = "",
  onchange
}) => {
  const checkboxId = id || `checkbox-${Math.random().toString(36).substring(2, 9)}`;

  return (
    <label
      htmlFor={checkboxId}
      className={`inline-flex items-start gap-2.5 cursor-pointer select-none ${
        disabled ? "opacity-50 cursor-not-allowed" : ""
      } ${className}`}
    >
      <div className="relative flex items-center justify-center mt-0.5">
        <input
          id={checkboxId}
          type="checkbox"
          checked={checked}
          disabled={disabled}
          onChange={(e) => onchange?.(e.target.checked)}
          className="sr-only peer"
        />
        <div
          className={`w-4 h-4 rounded-md border transition-colors flex items-center justify-center ${
            checked
              ? "bg-text border-text text-bg"
              : "bg-surface border-border hover:border-border-strong"
          }`}
        >
          {checked && <Icon name="check" size={11} />}
        </div>
      </div>

      {(label || description) && (
        <div className="flex flex-col text-left">
          {label && <span className="text-xs font-semibold text-text">{label}</span>}
          {description && <span className="text-[11px] text-text-subtle">{description}</span>}
        </div>
      )}
    </label>
  );
};
