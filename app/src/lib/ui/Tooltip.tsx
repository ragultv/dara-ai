import React, { useState } from "react";

export interface TooltipProps {
  content: string;
  children: React.ReactNode;
  position?: "top" | "bottom" | "left" | "right";
  className?: string;
}

export const Tooltip: React.FC<TooltipProps> = ({
  content,
  children,
  position = "top",
  className = ""
}) => {
  const [visible, setVisible] = useState(false);

  const positionClass =
    position === "bottom"
      ? "top-full left-1/2 -translate-x-1/2 mt-1.5"
      : position === "left"
        ? "right-full top-1/2 -translate-y-1/2 mr-1.5"
        : position === "right"
          ? "left-full top-1/2 -translate-y-1/2 ml-1.5"
          : "bottom-full left-1/2 -translate-x-1/2 mb-1.5";

  return (
    <div
      className={`relative inline-flex ${className}`}
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
      onFocus={() => setVisible(true)}
      onBlur={() => setVisible(false)}
    >
      {children}
      {visible && (
        <div
          role="tooltip"
          className={`absolute z-50 px-2 py-1 text-[11px] font-medium text-white bg-black/90 border border-white/10 rounded-md shadow-lg whitespace-nowrap pointer-events-none animate-in fade-in duration-150 ${positionClass}`}
        >
          {content}
        </div>
      )}
    </div>
  );
};
