import React from "react";
import type { ModelInfo } from "../../api/types.ts";
import { formatContext, formatMb, formatTokensExact } from "../../utils/format.ts";
import { Icon } from "../../ui/Icon.tsx";
import { fileRows, fileStateText } from "./modelHelpers.ts";

interface FilesProps {
  model: ModelInfo;
}

/** Model / Image support / Flash drafter: what is on disk, what can be added. */
export const ModelFiles: React.FC<FilesProps> = ({ model }) => (
  <ul className="rounded-md bg-surface-sunken border border-border divide-y divide-border text-xs">
    {fileRows(model).map((row) => (
      <li key={row.kind} className="flex items-center justify-between gap-2 px-3 py-2">
        <span className="font-medium text-text">{row.label}</span>
        <span className="flex items-center gap-1.5 text-text-muted">
          <Icon
            name={
              row.state === "installed" ? "check" : row.state === "available" ? "download" : "x"
            }
            size={12}
            className={row.state === "installed" ? "text-success" : "text-text-muted"}
          />
          <span>{fileStateText(row.state)}</span>
          {row.sizeMb !== null && row.state !== "unavailable" && (
            <span className="font-mono tabular-nums">({formatMb(row.sizeMb)})</span>
          )}
        </span>
      </li>
    ))}
  </ul>
);

interface LoadedProps {
  model: ModelInfo;
}

/** The configuration the server actually loaded (`dara.config`), never a guess. */
export const LoadedConfigSummary: React.FC<LoadedProps> = ({ model }) => {
  const cfg = model.dara.config;
  if (!model.dara.loaded || !cfg) return null;
  return (
    <div className="rounded-md bg-surface-raised border border-border p-3 space-y-1.5 text-xs">
      <p className="font-semibold text-text">Currently loaded</p>
      <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-text">
        <dt className="text-text-muted">Context</dt>
        <dd className="font-mono tabular-nums" title={formatTokensExact(cfg.context)}>
          {formatContext(cfg.context)} tokens
        </dd>
        <dt className="text-text-muted">K/V format</dt>
        <dd className="font-mono">{cfg.kv}</dd>
        <dt className="text-text-muted">Flash</dt>
        <dd>{cfg.flash ? "On" : "Off"}</dd>
        <dt className="text-text-muted">Image support</dt>
        <dd>{cfg.vision ? "On" : "Off"}</dd>
        <dt className="text-text-muted">On the GPU</dt>
        <dd className="font-mono tabular-nums">
          {cfg.layers_on_gpu} of {cfg.layers} layers, output head on the{" "}
          {cfg.head === "gpu" ? "GPU" : "CPU"}
        </dd>
      </dl>
      {cfg.reason && <p className="text-text-muted leading-snug">{cfg.reason}</p>}
    </div>
  );
};
