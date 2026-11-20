import React from "react";
import type { OptionEntry, OptionsResponse } from "../../api/types.ts";
import { formatContext, formatTokensExact } from "../../utils/format.ts";
import { Badge } from "../../ui/Badge.tsx";
import { Icon } from "../../ui/Icon.tsx";
import { fastestSetupText, kvMemoryText, placementText, speedText } from "./optionsLogic.ts";

interface Props {
  option: OptionEntry | undefined;
  recommended: OptionsResponse["recommended"] | null;
  isRecommended: boolean;
  onUseRecommended: (() => void) | null;
}

export const ResultCard: React.FC<Props> = ({
  option,
  recommended,
  isRecommended,
  onUseRecommended
}) => {
  if (!option) {
    return (
      <section
        aria-labelledby="result-title"
        className="rounded-2xl border border-border bg-surface p-4"
      >
        <h2 id="result-title" className="text-sm font-bold text-text">
          Result
        </h2>
        <p className="text-xs text-text-muted mt-1">
          The engine has no numbers for this combination.
        </p>
      </section>
    );
  }

  const fastSpeed = speedText(option.fast.speed);

  return (
    <section
      aria-labelledby="result-title"
      className="rounded-2xl border border-border bg-surface p-4 space-y-4"
    >
      <div className="space-y-1">
        <div className="flex flex-wrap items-center gap-2">
          <h2 id="result-title" className="text-sm font-bold text-text">
            {fastestSetupText(option.fast)}
          </h2>
          {isRecommended && (
            <Badge variant="primary" size="sm">
              <Icon name="check" size={10} className="mr-1" />
              Recommended
            </Badge>
          )}
        </div>
        {fastSpeed.isMeasured && fastSpeed.detail && (
          <p className="text-xs text-text-muted">{fastSpeed.detail}</p>
        )}
        <p className="text-xs text-text-muted">{placementText(option.fast)}.</p>
        {!option.fast.ok && (
          <p role="alert" className="text-xs text-danger">
            This setup does not fit your GPU.
          </p>
        )}
      </div>

      {recommended && (
        <div className="rounded-xl bg-surface-sunken border border-border p-3 text-xs space-y-1.5">
          <p className="font-semibold text-text">
            {isRecommended ? "Why this is recommended" : "What the engine recommends"}
          </p>
          <p className="text-text-muted leading-snug">{recommended.reason}</p>
          {!isRecommended && onUseRecommended && (
            <button
              type="button"
              onClick={onUseRecommended}
              className="text-link underline cursor-pointer focus-visible:outline-2 focus-visible:outline-focus"
            >
              Use the recommended setup
            </button>
          )}
        </div>
      )}

      <div className="overflow-x-auto">
        <table className="w-full text-xs text-left">
          <caption className="text-left text-sm font-bold text-text pb-2">Context ladder</caption>
          <thead>
            <tr className="text-text-muted border-b border-border">
              <th scope="col" className="py-1.5 pr-3 font-semibold">
                Context
              </th>
              <th scope="col" className="py-1.5 pr-3 font-semibold">
                Where it runs
              </th>
              <th scope="col" className="py-1.5 pr-3 font-semibold">
                Memory for the conversation
              </th>
              <th scope="col" className="py-1.5 font-semibold">
                Speed
              </th>
            </tr>
          </thead>
          <tbody>
            {option.points.map((p) => {
              const s = speedText(p.speed);
              return (
                <tr key={p.context} className="border-b border-border last:border-0 align-top">
                  <td
                    className="py-2 pr-3 font-mono tabular-nums text-text"
                    title={formatTokensExact(p.context)}
                  >
                    {formatContext(p.context)}
                  </td>
                  <td className="py-2 pr-3 text-text">
                    <span className="inline-flex items-center gap-1.5">
                      <Icon
                        name={!p.ok ? "x" : p.fully_on_gpu ? "check" : "alert-triangle"}
                        size={12}
                        className={
                          !p.ok ? "text-danger" : p.fully_on_gpu ? "text-success" : "text-warning"
                        }
                      />
                      {placementText(p)}
                    </span>
                  </td>
                  <td className="py-2 pr-3 font-mono tabular-nums text-text-muted">
                    {kvMemoryText(p)}
                  </td>
                  <td className="py-2 text-text">
                    <span className={s.isMeasured ? "font-mono tabular-nums" : "text-text-muted"}>
                      {s.main}
                    </span>
                    {s.detail && <span className="block text-text-muted">{s.detail}</span>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
};
