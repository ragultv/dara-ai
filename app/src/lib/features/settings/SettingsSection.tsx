import React from "react";
import { Icon } from "../../ui/Icon.tsx";

interface SettingsSectionProps {
  /** Anchor id; the in-page navigation scrolls to it. */
  id: string;
  title: string;
  description?: string;
  icon: string;
  children: React.ReactNode;
}

export const SettingsSection: React.FC<SettingsSectionProps> = ({
  id,
  title,
  description,
  icon,
  children
}) => (
  <section id={id} aria-labelledby={`${id}-title`} className="scroll-mt-6 space-y-3">
    <div className="space-y-0.5">
      <h2 id={`${id}-title`} className="flex items-center gap-2 text-base font-bold text-text">
        <Icon name={icon} size={16} className="text-primary" />
        {title}
      </h2>
      {description && <p className="text-xs text-text-muted">{description}</p>}
    </div>
    <div className="space-y-5 rounded-2xl border border-border bg-surface p-5">{children}</div>
  </section>
);
