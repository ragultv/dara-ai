import React from "react";

interface NumberTickerProps {
  value: number;
  decimalPlaces?: number;
  prefix?: string;
  suffix?: string;
  className?: string;
}

export const NumberTicker: React.FC<NumberTickerProps> = ({
  value = 0,
  decimalPlaces = 0,
  prefix = "",
  suffix = "",
  className = ""
}) => {
  const formatted =
    decimalPlaces > 0
      ? Number(value).toFixed(decimalPlaces)
      : Math.round(value).toLocaleString("en-US");

  return (
    <span
      className={`inline-flex items-center font-mono tabular-nums leading-none tracking-normal ${className}`}
      style={{ fontVariantNumeric: "tabular-nums", fontFeatureSettings: '"tnum" 1' }}
      aria-live="polite"
    >
      {prefix}
      {formatted}
      {suffix}
    </span>
  );
};
