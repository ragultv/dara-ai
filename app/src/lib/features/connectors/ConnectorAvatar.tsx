import React from "react";

/** The round-square badge before a connector's name: its first letter. (No brand logos are bundled.) */
export const ConnectorAvatar: React.FC<{ name: string; size?: "md" | "lg" }> = ({
  name,
  size = "md"
}) => (
  <span
    aria-hidden="true"
    className={`flex shrink-0 items-center justify-center rounded-xl border border-border bg-surface-raised font-[family-name:var(--font-reply)] font-semibold text-text ${
      size === "lg" ? "size-14 text-2xl" : "size-10 text-lg"
    }`}
  >
    {name.trim().charAt(0).toUpperCase() || "?"}
  </span>
);
