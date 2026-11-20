import React, { useState, useRef, useEffect } from "react";

interface PopoverProps {
  trigger: (props: { isOpen: boolean; toggle: () => void }) => React.ReactNode;
  children: (props: { close: () => void }) => React.ReactNode;
  side?: "top" | "bottom" | "left" | "right";
  className?: string;
}

export const Popover: React.FC<PopoverProps> = ({
  trigger,
  children,
  side = "top",
  className = ""
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    }
    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen]);

  const sideStyles = {
    top: "bottom-full left-0 mb-2",
    bottom: "top-full left-0 mt-2",
    left: "right-full top-0 mr-2",
    right: "left-full top-0 ml-2"
  };

  return (
    <div className="relative inline-block" ref={containerRef}>
      {trigger({ isOpen, toggle: () => setIsOpen((prev) => !prev) })}
      {isOpen && (
        <div
          className={`absolute ${sideStyles[side]} z-50 rounded-2xl border border-border bg-surface-raised p-3 shadow-xl animate-in fade-in zoom-in-95 duration-150 ${className}`}
        >
          {children({ close: () => setIsOpen(false) })}
        </div>
      )}
    </div>
  );
};
