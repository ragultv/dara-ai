import React from "react";
import { Icon } from "./Icon.tsx";

export interface EmptyStateProps {
  icon?: string;
  title: string;
  description?: string;
  action?: React.ReactNode;
  className?: string;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  icon = "inbox",
  title,
  description,
  action,
  className = ""
}) => {
  return (
    <div
      className={`flex flex-col items-center justify-center p-8 text-center space-y-3 rounded-2xl border border-dashed border-border bg-surface/30 ${className}`}
    >
      <div className="w-10 h-10 rounded-xl bg-surface-raised flex items-center justify-center text-text-subtle border border-border">
        <Icon name={icon} size={20} />
      </div>
      <div className="space-y-1 max-w-sm">
        <h3 className="text-sm font-bold text-text">{title}</h3>
        {description && <p className="text-xs text-text-muted leading-relaxed">{description}</p>}
      </div>
      {action && <div className="pt-2">{action}</div>}
    </div>
  );
};
