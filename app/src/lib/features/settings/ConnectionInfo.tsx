import React from "react";
import { useServerStore } from "../../stores/useServerStore.ts";
import { formatGb, formatBytes } from "../../utils/format.ts";
import { Badge } from "../../ui/Badge.tsx";
import { Button } from "../../ui/Button.tsx";
import { KeyValue } from "../monitor/KeyValue.tsx";
import { ApiKeyField } from "./ApiKeyField.tsx";
import type { ServerFacts } from "./useServerFacts.ts";

interface ConnectionInfoProps {
  facts: ServerFacts;
  onRegenerate?: (() => void) | undefined;
}

export const ConnectionInfo: React.FC<ConnectionInfoProps> = ({ facts, onRegenerate }) => {
  const baseUrl = useServerStore((s) => s.baseUrl);
  const apiKey = useServerStore((s) => s.apiKey);
  const status = useServerStore((s) => s.status);
  const { stats, version, store, storeError } = facts;

  let port: string | undefined;
  try {
    port = new URL(baseUrl).port || undefined;
  } catch {
    port = undefined;
  }

  const disk = stats?.disk;
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-sm font-semibold text-text">Connection</h3>
        <div className="flex items-center gap-2">
          <Badge variant={status === "ready" ? "success" : "danger"} size="sm">
            {status === "ready" ? "Engine running" : "Engine not reachable"}
          </Badge>
          <Button variant="ghost" size="xs" onClick={facts.refresh}>
            Refresh
          </Button>
        </div>
      </div>

      <KeyValue
        rows={[
          { label: "Address", value: baseUrl },
          { label: "Port", value: port },
          { label: "Engine version", value: version?.version },
          { label: "API version", value: version ? String(version.api) : undefined },
          { label: "Engine build", value: version?.built },
          { label: "Loaded model", value: stats ? (stats.model?.id ?? "None") : undefined },
          { label: "Models folder", value: disk?.models_dir, title: disk?.models_dir },
          {
            label: "Free space on that drive",
            value: disk
              ? `${formatGb(disk.volume_free_gb)} of ${formatGb(disk.volume_total_gb)}`
              : undefined
          }
        ]}
      />

      <ApiKeyField apiKey={apiKey} onRegenerate={onRegenerate} />

      <div className="space-y-2">
        <h3 className="text-sm font-semibold text-text">Database</h3>
        {store ? (
          <KeyValue
            rows={[
              { label: "File", value: store.path, title: store.path },
              { label: "Size", value: formatBytes(store.size_bytes) },
              { label: "Schema version", value: String(store.schema_version) },
              { label: "Conversations", value: String(store.conversations) },
              { label: "Messages", value: String(store.messages) },
              { label: "Images", value: String(store.attachments) },
              { label: "Saved settings", value: String(store.settings) }
            ]}
          />
        ) : (
          <p role="status" className="text-xs text-text-muted">
            {storeError ?? "Reading the database..."}
          </p>
        )}
      </div>
    </div>
  );
};
