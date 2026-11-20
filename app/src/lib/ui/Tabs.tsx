import React from "react";

export interface TabItem {
  id: string;
  label: string;
  count?: number;
  disabled?: boolean;
}

export interface TabsProps {
  tabs: TabItem[];
  activeId: string;
  onchange: (id: string) => void;
  className?: string;
}

export const Tabs: React.FC<TabsProps> = ({ tabs, activeId, onchange, className = "" }) => {
  return (
    <div className={`flex border-b border-border space-x-1 ${className}`} role="tablist">
      {tabs.map((tab) => {
        const isActive = tab.id === activeId;
        return (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={isActive}
            disabled={tab.disabled}
            onClick={() => onchange(tab.id)}
            className={`flex items-center gap-2 px-3.5 py-2 text-xs font-semibold border-b-2 -mb-px transition-colors cursor-pointer ${
              isActive
                ? "border-border-strong text-text"
                : "border-transparent text-text-muted hover:text-text hover:border-border"
            } ${tab.disabled ? "opacity-40 cursor-not-allowed" : ""}`}
          >
            <span>{tab.label}</span>
            {tab.count !== undefined && (
              <span className="px-1.5 py-0.2 rounded-full text-[10px] bg-surface-raised border border-border text-text-subtle">
                {tab.count}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
};
