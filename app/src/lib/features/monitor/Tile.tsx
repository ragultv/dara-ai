import React from "react";
import { Icon } from "../../ui/Icon.tsx";

interface TileProps {
  label: string;
  value: string;
  sub?: string | undefined;
  icon?: string | undefined;
  /** Exact value on hover. */
  title?: string | undefined;
}

/** A summary number at the top of the monitor. */
export const Tile: React.FC<TileProps> = ({ label, value, sub, icon, title }) => (
  <div className="min-w-0 rounded-2xl border border-border bg-surface p-4 text-left" title={title}>
    <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-text-muted">
      {icon && <Icon name={icon} size={12} className="text-primary" />}
      <span className="truncate">{label}</span>
    </div>
    <div className="mt-1.5 truncate text-lg font-bold tabular-nums text-text">{value}</div>
    {sub && <div className="mt-0.5 truncate text-[11px] text-text-muted">{sub}</div>}
  </div>
);
