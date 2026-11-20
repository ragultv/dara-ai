import React from "react";

export interface KeyValueRow {
  label: string;
  /** Rows with an undefined value are not rendered: the API did not return it. */
  value: React.ReactNode | undefined;
  /** Exact value on hover. */
  title?: string | undefined;
}

interface KeyValueProps {
  rows: KeyValueRow[];
}

export const KeyValue: React.FC<KeyValueProps> = ({ rows }) => {
  const shown = rows.filter((r) => r.value !== undefined && r.value !== null);
  if (shown.length === 0) return null;
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-xs">
      {shown.map((r) => (
        <React.Fragment key={r.label}>
          <dt className="text-text-muted">{r.label}</dt>
          <dd
            className="min-w-0 truncate text-right font-medium tabular-nums text-text"
            title={r.title}
          >
            {r.value}
          </dd>
        </React.Fragment>
      ))}
    </dl>
  );
};
