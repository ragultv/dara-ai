import React from "react";
import { Icon } from "./Icon.tsx";

export interface IconButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  icon: string;
  size?: "sm" | "md" | "lg";
  variant?: "ghost" | "secondary" | "primary" | "danger";
  label: string;
}

export const IconButton: React.FC<IconButtonProps> = ({
  icon,
  size = "md",
  variant = "ghost",
  label,
  className = "",
  disabled,
  ...rest
}) => {
  const sizeClasses =
    size === "sm" ? "w-7 h-7 text-xs" : size === "lg" ? "w-10 h-10 text-base" : "w-8 h-8 text-sm";

  const iconSizes = size === "sm" ? 13 : size === "lg" ? 18 : 15;

  const variantClasses =
    variant === "primary"
      ? "bg-primary text-white hover:bg-primary-hover shadow-xs"
      : variant === "secondary"
        ? "bg-surface border border-border text-text hover:bg-surface-raised"
        : variant === "danger"
          ? "text-danger hover:bg-danger/10"
          : "text-text-subtle hover:text-text hover:bg-surface-raised";

  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      disabled={disabled}
      className={`inline-flex items-center justify-center rounded-lg transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed ${sizeClasses} ${variantClasses} ${className}`}
      {...rest}
    >
      <Icon name={icon} size={iconSizes} />
    </button>
  );
};
