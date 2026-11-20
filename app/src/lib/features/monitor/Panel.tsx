import React from "react";
import { Icon } from "../../ui/Icon.tsx";

interface PanelProps {
  title: string;
  icon?: string;
  /** Right-aligned content in the header (a badge, a status). */
  aside?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}

/** A titled card of the monitor dashboard. */
export const Panel: React.FC<PanelProps> = ({ title, icon, aside, children, className = "" }) => (
  <section className={`rounded-2xl border border-border bg-surface p-4 text-left ${className}`}>
    <header className="mb-3 flex items-center justify-between gap-2">
      <h2 className="flex items-center gap-2 text-sm font-semibold text-text">
        {icon && <Icon name={icon} size={15} className="text-text" />}
        {title}
      </h2>
      {aside}
    </header>
    <div className="space-y-3">{children}</div>
  </section>
);
