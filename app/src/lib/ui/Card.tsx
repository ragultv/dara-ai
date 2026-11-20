import React from "react";

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  className?: string;
}

export const Card: React.FC<CardProps> = ({ children, className = "", ...props }) => {
  return (
    <div
      className={`rounded-2xl border border-border bg-surface p-4 shadow-xs transition-colors ${className}`}
      {...props}
    >
      {children}
    </div>
  );
};
