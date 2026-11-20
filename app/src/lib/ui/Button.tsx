import React from "react";

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "ghost" | "danger";
  size?: "xs" | "sm" | "md" | "lg";
  children?: React.ReactNode;
}

export const Button: React.FC<ButtonProps> = ({
  variant = "secondary",
  size = "md",
  className = "",
  children,
  ...props
}) => {
  const base =
    "inline-flex items-center justify-center font-medium rounded-xl transition-all cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed select-none outline-none focus-visible:ring-2 focus-visible:ring-primary/40";

  const sizeClasses = {
    xs: "h-7 px-2.5 text-xs gap-1",
    sm: "h-8 px-3 text-xs gap-1.5",
    md: "h-9 px-4 text-xs font-semibold gap-2",
    lg: "h-11 px-5 text-sm font-semibold gap-2.5"
  };

  const variantClasses = {
    primary: "bg-primary hover:bg-primary-hover text-white shadow-sm active:scale-[0.98]",
    secondary: "bg-surface-raised hover:bg-surface-sunken text-text border border-border shadow-xs",
    ghost: "bg-transparent hover:bg-surface-raised text-text-muted hover:text-text",
    danger: "bg-danger hover:bg-danger-hover text-white shadow-sm"
  };

  return (
    <button
      className={`${base} ${sizeClasses[size]} ${variantClasses[variant]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
};
