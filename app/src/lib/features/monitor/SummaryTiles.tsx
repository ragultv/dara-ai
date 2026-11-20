import React from "react";
import type { Stats } from "../../api/types.ts";
import { formatContext, formatInt, formatUptime } from "../../utils/format.ts";
import { Tile } from "./Tile.tsx";
import { isFlashLoaded, readString } from "./model-info.ts";

interface SummaryTilesProps {
  stats: Stats;
}

export const SummaryTiles: React.FC<SummaryTilesProps> = ({ stats }) => {
  const model = stats.model;
  const quant = readString(model, "quant");
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-5">
      {model ? (
        <>
          <Tile
            icon="box"
            label="Loaded model"
            value={model.id}
            sub={[quant, `K/V ${model.kv_format}`].filter(Boolean).join(" · ")}
          />
          <Tile
            icon="layers"
            label="Context"
            value={`${formatInt(stats.context.held_tokens)} / ${formatContext(stats.context.capacity)}`}
            sub="tokens held / capacity"
            title={`${stats.context.held_tokens.toLocaleString("en-US")} of ${stats.context.capacity.toLocaleString("en-US")} tokens`}
          />
          <Tile
            icon="cpu"
            label="Layers on GPU"
            value={`${model.layers_on_gpu} / ${model.layers}`}
            sub={`Output head on ${model.output_head === "gpu" ? "GPU" : "CPU"}`}
          />
          <Tile
            icon="zap"
            label="Flash"
            value={isFlashLoaded(model) ? "On" : "Off"}
            sub={isFlashLoaded(model) ? "Drafter loaded" : "No drafter loaded"}
          />
        </>
      ) : (
        <Tile
          icon="box"
          label="Loaded model"
          value="None"
          sub="Load a model from the Models page"
        />
      )}
      <Tile icon="clock" label="Engine uptime" value={formatUptime(stats.uptime_seconds)} />
    </div>
  );
};
