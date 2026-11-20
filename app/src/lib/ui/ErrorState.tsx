import React from "react";
import { Icon } from "./Icon.tsx";

export interface ErrorStateProps {
  title?: string;
  message: string;
  onRetry?: () => void;
  className?: string;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  title = "Something went wrong",
  message,
  onRetry,
  className = ""
}) => {
  return (
    <div
      role="alert"
      className={`flex flex-col items-center justify-center p-6 text-center space-y-3 rounded-2xl border border-danger/20 bg-danger/5 ${className}`}
    >
      <div className="w-10 h-10 rounded-xl bg-danger/10 flex items-center justify-center text-danger border border-danger/20">
        <Icon name="alert-triangle" size={20} />
      </div>
      <div className="space-y-1 max-w-sm">
        <h3 className="text-sm font-bold text-danger">{title}</h3>
        <p className="text-xs text-text-muted leading-relaxed">{message}</p>
      </div>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="px-3 py-1.5 rounded-lg border border-border bg-surface hover:bg-surface-raised text-xs font-semibold text-text transition-colors cursor-pointer"
        >
          Try Again
        </button>
      )}
    </div>
  );
};
