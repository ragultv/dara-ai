import React from "react";

export interface SliderProps {
  label: string;
  min: number;
  max: number;
  step?: number;
  value: number;
  unit?: string;
  onChange?: (val: number) => void;
  oninput?: (val: number) => void;
  className?: string;
}

export const Slider: React.FC<SliderProps> = ({
  label,
  min,
  max,
  step = 1,
  value,
  unit = "",
  onChange,
  oninput,
  className = ""
}) => {
  const handler = onChange || oninput;

  return (
    <div className={`space-y-1.5 text-left ${className}`}>
      <div className="flex items-center justify-between text-xs font-medium text-text">
        <span>{label}</span>
        <span className="font-mono text-text-muted">
          {value}
          {unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => handler?.(Number(e.target.value))}
        className="w-full accent-primary h-1.5 bg-surface-sunken rounded-lg cursor-pointer"
      />
    </div>
  );
};
