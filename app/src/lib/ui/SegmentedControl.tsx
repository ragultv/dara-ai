import React from "react";

export interface SegmentOption {
  value: string;
  label: string;
  description?: string;
  disabled?: boolean;
}

export interface SegmentedControlProps {
  id?: string;
  label?: string;
  options: SegmentOption[];
  value?: string;
  disabled?: boolean;
  className?: string;
  onchange?: (value: string) => void;
}

export const SegmentedControl: React.FC<SegmentedControlProps> = ({
  id,
  label,
  options,
  value,
  disabled = false,
  className = "",
  onchange
}) => {
  const controlId = id || `segment-${Math.random().toString(36).substring(2, 9)}`;
  const activeValue = value !== undefined ? value : (options[0]?.value ?? "");

  const handleSelect = (val: string) => {
    if (disabled) return;
    onchange?.(val);
  };

  return (
    <div className={`flex flex-col space-y-1.5 w-full text-left ${className}`}>
      {label && (
        <span id={`${controlId}-label`} className="text-xs font-semibold text-text">
          {label}
        </span>
      )}

      <div
        role="radiogroup"
        aria-labelledby={label ? `${controlId}-label` : undefined}
        className="flex p-1 rounded-xl bg-surface-sunken border border-border w-full gap-1"
      >
        {options.map((opt) => {
          const isSelected = opt.value === activeValue;
          return (
            <button
              key={opt.value}
              type="button"
              role="radio"
              aria-checked={isSelected}
              disabled={disabled || opt.disabled}
              onClick={() => handleSelect(opt.value)}
              className={`flex-1 py-1.5 px-3 text-center text-xs font-semibold rounded-lg transition-all cursor-pointer truncate ${
                isSelected
                  ? "bg-surface text-text shadow-xs font-bold border border-border-strong"
                  : "text-text-muted hover:text-text hover:bg-surface/50 border border-transparent"
              } ${disabled || opt.disabled ? "opacity-40 cursor-not-allowed" : ""}`}
            >
              {opt.label}
            </button>
          );
        })}
      </div>
    </div>
  );
};
