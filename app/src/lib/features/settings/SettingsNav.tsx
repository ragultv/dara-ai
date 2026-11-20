import React from "react";

export interface NavItem {
  id: string;
  label: string;
}

interface SettingsNavProps {
  items: NavItem[];
  active: string;
  onSelect: (id: string) => void;
}

/** Settings tabs: one section is shown at a time (the page renders only the active one). */
export const SettingsNav: React.FC<SettingsNavProps> = ({ items, active, onSelect }) => (
  <div
    role="tablist"
    aria-label="Settings sections"
    aria-orientation="vertical"
    className="flex flex-col gap-1"
  >
    {items.map((item) => {
      const selected = active === item.id;
      return (
        <button
          key={item.id}
          type="button"
          role="tab"
          id={`tab-${item.id}`}
          aria-selected={selected}
          aria-controls={`panel-${item.id}`}
          tabIndex={selected ? 0 : -1}
          onClick={() => onSelect(item.id)}
          className={`w-full cursor-pointer rounded-lg px-3 py-2 text-left text-xs font-semibold transition-colors ${
            selected
              ? "bg-surface-raised text-text"
              : "text-text-muted hover:bg-surface-raised hover:text-text"
          }`}
        >
          {item.label}
        </button>
      );
    })}
  </div>
);
