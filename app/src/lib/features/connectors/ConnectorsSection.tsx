import React, { useEffect, useState } from "react";
import { useConnectorsStore } from "../../stores/useConnectorsStore.ts";
import { AddConnectorDialog } from "../../connectors/AddConnectorDialog.tsx";
import { getBackend } from "../../connectors/backend.ts";
import { CatalogList } from "./CatalogList.tsx";
import { ConnectorDetail } from "./ConnectorDetail.tsx";

/** Settings > Connectors: the apps view, and one connector's page when you open it. */
export const ConnectorsSection: React.FC = () => {
  const load = useConnectorsStore((s) => s.load);
  const entries = useConnectorsStore((s) => s.entries);
  const [openKey, setOpenKey] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const isDemo = getBackend().isDemo;

  useEffect(() => {
    void load();
  }, [load]);

  // a connector added by hand opens its own page once it exists
  const [pendingCustom, setPendingCustom] = useState<number | null>(null);
  useEffect(() => {
    if (pendingCustom === null || entries.length <= pendingCustom) return;
    const newest = [...entries].sort((a, b) => b.config.createdAt - a.config.createdAt)[0];
    if (newest) setOpenKey(`c:${newest.config.id}`);
    setPendingCustom(null);
  }, [entries, pendingCustom]);

  return (
    <div>
      {isDemo && (
        <p className="mb-5 rounded-lg border border-border bg-surface-raised px-4 py-3 text-xs text-text-muted">
          You are looking at a demo with sample data. Connectors run in the Dara desktop app.
        </p>
      )}
      {openKey ? (
        <ConnectorDetail itemKey={openKey} onBack={() => setOpenKey(null)} />
      ) : (
        <CatalogList
          onOpen={setOpenKey}
          onAddCustom={() => {
            setPendingCustom(entries.length);
            setAdding(true);
          }}
        />
      )}
      <AddConnectorDialog
        isOpen={adding}
        onClose={() => {
          setAdding(false);
        }}
      />
    </div>
  );
};
