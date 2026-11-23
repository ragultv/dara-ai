import React, { useEffect, useState } from "react";
import { useConnectorsStore } from "../../stores/useConnectorsStore.ts";
import type { CatalogEntry } from "../../connectors/catalog.ts";
import type { ConnectorEntry } from "../../connectors/types.ts";
import { checkApiKey } from "../../connectors/validate.ts";
import { Button } from "../../ui/Button.tsx";
import { Dialog } from "../../ui/Dialog.tsx";
import { Input } from "../../ui/Input.tsx";
import { addFromCatalog } from "./addFromCatalog.ts";

const linkButton =
  "cursor-pointer text-xs text-text-muted underline underline-offset-2 hover:text-text";

/** Opens a page in the system browser (the desktop app), or a new tab (a browser). */
function openPage(url: string) {
  void import("@tauri-apps/plugin-opener")
    .then((m) => m.openUrl(url))
    .catch(() => window.open(url, "_blank", "noopener,noreferrer"));
}

const Steps: React.FC<{ entry?: CatalogEntry | undefined }> = ({ entry }) =>
  entry?.keyHelp?.length || entry?.keyUrl ? (
    <div className="space-y-1.5">
      {entry.keyHelp?.length ? (
        <ol className="list-decimal space-y-1 pl-5 text-xs leading-relaxed text-text-muted">
          {entry.keyHelp.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ol>
      ) : null}
      {entry.keyUrl && (
        <button type="button" onClick={() => openPage(entry.keyUrl as string)} className={linkButton}>
          Get a key
        </button>
      )}
    </div>
  ) : null;

/** Asks for the API key of a catalog connector that has no sign-in, then adds it. */
export const ApiKeyDialog: React.FC<{
  entry: CatalogEntry;
  isOpen: boolean;
  onClose: () => void;
  onAdded: () => void;
}> = ({ entry, isOpen, onClose, onAdded }) => {
  const [key, setKey] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const close = () => {
    setKey("");
    setError(null);
    onClose();
  };
  const submit = async () => {
    const bad = checkApiKey(key, entry.keyHeader);
    if (bad) return setError(bad);
    setBusy(true);
    const problem = await addFromCatalog(entry, key);
    setBusy(false);
    if (problem) setError(problem);
    else {
      setKey("");
      setError(null);
      onAdded();
    }
  };

  return (
    <Dialog
      isOpen={isOpen}
      title={`Add ${entry.name}`}
      description={`${entry.name} does not use a sign-in. Paste your API key; it is kept in the Windows credential store and sent only to ${new URL(entry.url).host}.`}
      onclose={close}
      actions={
        <>
          <Button variant="ghost" size="sm" onClick={close}>
            Cancel
          </Button>
          <Button variant="primary" size="sm" disabled={busy} onClick={() => void submit()}>
            Add connector
          </Button>
        </>
      }
    >
      <div className="space-y-3 text-left">
        <Steps entry={entry} />
        <Input
          label="API key"
          type="password"
          value={key}
          onChange={(e) => setKey(e.target.value)}
          autoComplete="off"
          spellCheck={false}
          autoFocus
        />
        {error && (
          <p role="alert" className="text-xs text-danger">
            {error}
          </p>
        )}
      </div>
    </Dialog>
  );
};

/**
 * The "API key" line of a connector's details: whether a key is saved (never the key itself), and how to replace or remove it. For a
 * custom server without a key it offers to use one instead of signing in.
 */
export const ApiKeyRow: React.FC<{
  connector: ConnectorEntry;
  entry?: CatalogEntry | undefined;
  Row: React.FC<{ label: string; children: React.ReactNode }>;
}> = ({ connector, entry, Row }) => {
  const update = useConnectorsStore((s) => s.update);
  const connect = useConnectorsStore((s) => s.connect);
  const { hasKey, header } = connector.config.apiKey;
  const id = connector.config.id;
  const keyOnly = entry?.registers === "api-key";
  // a server that has only an API key opens the form straight away until a key is saved
  const [editing, setEditing] = useState(keyOnly && !hasKey);
  const [key, setKey] = useState("");
  const [newHeader, setNewHeader] = useState("");
  const [error, setError] = useState<string | null>(null);

  // The catalog says which header this server wants. A connector added before that was known (or with another header) is corrected
  // here without asking for the key again.
  const wantedHeader = keyOnly ? entry?.keyHeader : undefined;
  const headerWrong = hasKey && wantedHeader !== undefined && header !== wantedHeader;
  useEffect(() => {
    if (headerWrong && wantedHeader) void update(id, { apiKey: { header: wantedHeader } });
  }, [headerWrong, wantedHeader, id, update]);

  if (connector.config.kind !== "http") return null;
  // a catalog connector that signs in has no use for a key
  if (entry && !keyOnly && !hasKey) return null;

  const stop = () => {
    setEditing(false);
    setKey("");
    setNewHeader("");
    setError(null);
  };
  const save = async () => {
    const h = keyOnly ? entry?.keyHeader : newHeader.trim() || header || undefined;
    const bad = checkApiKey(key, h ?? undefined);
    if (bad) return setError(bad);
    const problem = await update(id, {
      apiKey: { value: key, ...(h ? { header: h } : {}) }
    });
    if (problem) return setError(problem);
    stop();
    void connect(id);
  };
  const remove = async () => {
    const problem = await update(id, { apiKey: { clear: true } });
    if (problem) setError(problem);
  };

  if (!editing && !hasKey)
    return (
      <Row label="API key">
        <span className="text-text-muted">None.</span>{" "}
        <button type="button" onClick={() => setEditing(true)} className={linkButton}>
          Use an API key instead of signing in
        </button>
      </Row>
    );

  if (!editing)
    return (
      <Row label="API key">
        Saved on this computer
        <span className="block text-xs text-text-muted">
          Sent as {header ? `${header}: <key>` : "Authorization: Bearer <key>"}
        </span>
        <span className="mt-1 flex gap-3">
          <button type="button" onClick={() => setEditing(true)} className={linkButton}>
            Replace
          </button>
          {!keyOnly && (
            <button type="button" onClick={() => void remove()} className={linkButton}>
              Remove
            </button>
          )}
        </span>
        {error && (
          <span role="alert" className="mt-1 block text-xs text-danger">
            {error}
          </span>
        )}
      </Row>
    );

  return (
    <div className="my-3 rounded-lg border border-border bg-surface-raised p-4">
      <p className="text-sm font-semibold text-text">
        {hasKey ? "Replace the API key" : "Use an API key"}
      </p>
      <div className="mt-2">
        <Steps entry={entry} />
      </div>
      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        <Input
          label="API key"
          type="password"
          value={key}
          onChange={(e) => setKey(e.target.value)}
          autoComplete="off"
          spellCheck={false}
        />
        {!keyOnly && (
          <Input
            label="Header (optional)"
            hint="Empty: Authorization: Bearer <key>"
            placeholder={header ?? "X-API-Key"}
            value={newHeader}
            onChange={(e) => setNewHeader(e.target.value)}
            spellCheck={false}
          />
        )}
      </div>
      {error && (
        <p role="alert" className="mt-2 text-xs text-danger">
          {error}
        </p>
      )}
      <div className="mt-3 flex gap-2">
        <Button variant="primary" size="sm" onClick={() => void save()}>
          Save
        </Button>
        {(hasKey || !keyOnly) && (
          <Button variant="ghost" size="sm" onClick={stop}>
            Cancel
          </Button>
        )}
      </div>
    </div>
  );
};
