import React, { useEffect, useState } from "react";
import { Input } from "../../ui/Input.tsx";

interface NumberFieldProps {
  label: string;
  hint?: string;
  value: number;
  min: number;
  max: number;
  disabled?: boolean;
  /** Called only with a valid whole number inside [min, max]. */
  onCommit: (value: number) => void;
}

/** Keeps the typed text while editing, so an empty or half-typed field is never forced to a default. */
export const NumberField: React.FC<NumberFieldProps> = ({
  label,
  hint,
  value,
  min,
  max,
  disabled = false,
  onCommit
}) => {
  const [text, setText] = useState(String(value));

  useEffect(() => {
    setText(String(value));
  }, [value]);

  const parsed = Number(text);
  const isValid = text.trim() !== "" && Number.isInteger(parsed) && parsed >= min && parsed <= max;

  return (
    <Input
      label={label}
      type="number"
      inputMode="numeric"
      min={min}
      max={max}
      value={text}
      disabled={disabled}
      {...(hint !== undefined ? { hint } : {})}
      {...(!isValid ? { error: `Enter a whole number from ${min} to ${max}.` } : {})}
      onChange={(e) => {
        setText(e.target.value);
        const n = Number(e.target.value);
        if (e.target.value.trim() !== "" && Number.isInteger(n) && n >= min && n <= max)
          onCommit(n);
      }}
    />
  );
};
