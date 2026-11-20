import React from "react";

export interface ProgressBarProps {
  value: number; // 0 to 100
  label?: string;
  variant?: "primary" | "success" | "warning" | "danger";
  size?: "sm" | "md" | "lg";
  className?: string;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({
  value,
  label,
  variant = "primary",
  size = "md",
  className = ""
}) => {
  const clamped = Math.min(100, Math.max(0, value));

  const heightClass = size === "sm" ? "h-1.5" : size === "lg" ? "h-3" : "h-2";

  const colorClass =
    variant === "success"
      ? "bg-success"
      : variant === "warning"
        ? "bg-warning"
        : variant === "danger"
          ? "bg-danger"
          : "bg-text";

  return (
    <div className={`w-full space-y-1.5 ${className}`}>
      {label && (
        <div className="flex justify-between text-xs font-medium text-text-subtle">
          <span>{label}</span>
          <span className="font-mono">{Math.round(clamped)}%</span>
        </div>
      )}
      <div
        className={`w-full ${heightClass} rounded-full bg-surface-sunken overflow-hidden border border-border/40`}
      >
        <div
          className={`${heightClass} ${colorClass} rounded-full transition-all duration-300 ease-out`}
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  );
};
