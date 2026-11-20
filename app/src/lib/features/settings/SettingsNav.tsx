import React, { useEffect, useState } from "react";

export interface NavItem {
  id: string;
  label: string;
}

interface SettingsNavProps {
  items: NavItem[];
}

/** Sticky in-page navigation. Buttons scroll to the section; the one in view is highlighted. */
export const SettingsNav: React.FC<SettingsNavProps> = ({ items }) => {
  const [active, setActive] = useState<string>(items[0]?.id ?? "");

  useEffect(() => {
    if (typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting);
        const top = visible.sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
        if (top) setActive(top.target.id);
      },
      { rootMargin: "-10% 0px -70% 0px" }
    );
    for (const item of items) {
      const el = document.getElementById(item.id);
      if (el) observer.observe(el);
    }
    return () => observer.disconnect();
  }, [items]);

  const go = (id: string): void => {
    setActive(id);
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <nav
      aria-label="Settings sections"
      className="sticky top-0 hidden w-40 shrink-0 self-start min-[900px]:block"
    >
      <ul className="space-y-1">
        {items.map((item) => (
          <li key={item.id}>
            <button
              type="button"
              onClick={() => go(item.id)}
              aria-current={active === item.id ? "true" : undefined}
              className={`w-full cursor-pointer rounded-lg px-3 py-2 text-left text-xs font-semibold transition-colors ${
                active === item.id
                  ? "bg-primary-subtle text-on-primary-subtle"
                  : "text-text-muted hover:bg-surface-raised hover:text-text"
              }`}
            >
              {item.label}
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
};
