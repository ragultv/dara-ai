import React from "react";
import { ICONS } from "./icon-map.ts";

export type IconName =
  | "zap"
  | "sparkles"
  | "cpu"
  | "activity"
  | "settings"
  | "sliders"
  | "sliders-horizontal"
  | "message-square"
  | "message-square-plus"
  | "plus"
  | "send"
  | "square"
  | "refresh-cw"
  | "copy"
  | "check"
  | "thumbs-up"
  | "thumbs-down"
  | "pencil"
  | "trash-2"
  | "chevron-down"
  | "chevron-right"
  | "chevron-up"
  | "download"
  | "x"
  | "alert-triangle"
  | "info"
  | "terminal"
  | "shield"
  | "monitor"
  | "hard-drive"
  | "box"
  | "sun"
  | "moon"
  | "laptop";

interface IconProps extends React.SVGProps<SVGSVGElement> {
  name: IconName | string;
  size?: number | string;
  className?: string;
}

// Icons come from a generated map of only the lucide icons the source names (tools/gen-icons.mjs, run by `prebuild`), not the whole library.
export const Icon: React.FC<IconProps> = ({ name, size = 16, className = "", ...rest }) => {
  const Component = ICONS[name];

  if (!Component) {
    return (
      <svg
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className={className}
        {...rest}
      >
        <circle cx="12" cy="12" r="10" />
      </svg>
    );
  }

  return <Component size={size} className={className} {...rest} />;
};
