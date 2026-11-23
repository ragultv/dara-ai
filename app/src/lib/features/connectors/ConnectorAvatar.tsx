import React from "react";
import { logoForName } from "../../connectors/catalog.ts";

/** The round-square badge before a connector's name: its logo, or its first letter when it has none. */
export const ConnectorAvatar: React.FC<{ name: string; size?: "md" | "lg" }> = ({
  name,
  size = "md"
}) => {
  const logo = logoForName(name);
  return (
    <span
      aria-hidden="true"
      className={`flex shrink-0 items-center justify-center rounded-xl border border-border bg-white font-[family-name:var(--font-reply)] font-semibold text-text ${
        size === "lg" ? "size-14 text-2xl" : "size-10 text-lg"
      }`}
    >
      {logo ? (
        <img src={logo} alt="" className={size === "lg" ? "size-8" : "size-5"} draggable={false} />
      ) : (
        name.trim().charAt(0).toUpperCase() || "?"
      )}
    </span>
  );
};
