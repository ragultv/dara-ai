import React, { useEffect, useRef, useState } from "react";
import { useConnectorsStore } from "../../stores/useConnectorsStore.ts";
import {
  builtinClient,
  hostOf,
  humanizeToolName,
  type CatalogEntry
} from "../../connectors/catalog.ts";
import type { ConnectorEntry } from "../../connectors/types.ts";
import { relativeTime } from "../../artifacts/format.ts";
import { Icon } from "../../ui/Icon.tsx";
import { Button } from "../../ui/Button.tsx";
import { Dialog } from "../../ui/Dialog.tsx";
import { Input } from "../../ui/Input.tsx";
import { Skeleton } from "../../ui/Skeleton.tsx";
import { toast } from "../../ui/Toast.tsx";
import { ConnectorAvatar } from "./ConnectorAvatar.tsx";
import { PermissionsTab } from "./PermissionsTab.tsx";
import { addFromCatalog } from "./addFromCatalog.ts";
import { buildItems, statusText, type ConnectorItem } from "./items.ts";

type Tab = "overview" | "permissions";

const Row: React.FC<{ label: string; children: React.ReactNode }> = ({ label, children }) => (
  <div className="flex gap-6 py-2 text-sm">
    <dt className="w-32 shrink-0 text-text-muted">{label}</dt>
    <dd className="min-w-0 flex-1 text-text">{children}</dd>
  </div>
);

const Side: React.FC<{ label: string; children: React.ReactNode }> = ({ label, children }) => (
  <div>
    <p className="text-xs text-text-subtle">{label}</p>
    <div className="mt-1 text-sm text-text">{children}</div>
  </div>
);

/** Google-style servers have no registration: the user brings a client ID from the provider's console. */
const ClientForm: React.FC<{ entry: CatalogEntry; connector: ConnectorEntry }> = ({
  entry,
  connector
}) => {
  const update = useConnectorsStore((s) => s.update);
  const have = connector.config.oauth.clientId;
  const builtin = builtinClient(entry);
  const usingBuiltin = !!builtin && have === builtin.clientId;
  const [editing, setEditing] = useState(!have);
  const [clientId, setClientId] = useState("");
  const [secret, setSecret] = useState("");
  const [error, setError] = useState<string | null>(null);

  // Dara's own client: nothing to set up; a user who wants their own can still switch
  if (usingBuiltin && !editing)
    return (
      <Row label="Sign-in app">
        Dara's {entry.publisher} sign-in{" "}
        <button
          type="button"
          onClick={() => setEditing(true)}
          className="cursor-pointer text-xs text-text-muted underline underline-offset-2 hover:text-text"
        >
          Use my own
        </button>
      </Row>
    );
  if (have && !editing)
    return (
      <Row label="Client ID">
        <span className="break-all font-mono text-xs">{have}</span>{" "}
        <button
          type="button"
          onClick={() => setEditing(true)}
          className="cursor-pointer text-xs text-text-muted underline underline-offset-2 hover:text-text"
        >
          Change
        </button>
      </Row>
    );

  const save = async () => {
    if (!clientId.trim()) return setError("Paste the client ID first.");
    const problem = await update(connector.config.id, {
      oauth: { clientId: clientId.trim(), clientSecret: secret }
    });
    if (problem) setError(problem);
    else {
      setError(null);
      setEditing(false);
      setClientId("");
      setSecret("");
    }
  };
  return (
    <div className="my-3 rounded-lg border border-border bg-surface-raised p-4">
      <p className="text-sm font-semibold text-text">{entry.publisher} needs your own client ID</p>
      <ol className="mt-2 list-decimal space-y-1 pl-5 text-xs leading-relaxed text-text-muted">
        {(entry.clientHelp ?? []).map((l) => (
          <li key={l}>{l}</li>
        ))}
      </ol>
      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        <Input
          label="Client ID"
          value={clientId}
          onChange={(e) => setClientId(e.target.value)}
          spellCheck={false}
        />
        <Input
          label="Client secret (if shown)"
          type="password"
          value={secret}
          onChange={(e) => setSecret(e.target.value)}
          autoComplete="off"
        />
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
        {have && (
          <Button variant="ghost" size="sm" onClick={() => setEditing(false)}>
            Cancel
          </Button>
        )}
      </div>
    </div>
  );
};

const AccessLevel: React.FC<{ entry: CatalogEntry; connector: ConnectorEntry }> = ({
  entry,
  connector
}) => {
  const update = useConnectorsStore((s) => s.update);
  const levels = entry.access;
  if (!levels?.length) return null;
  const current = levels.find((l) => l.scope === connector.config.oauth.scope) ?? levels[0]!;
  return (
    <Row label="Access">
      <div role="radiogroup" aria-label="Access level" className="space-y-2">
        {levels.map((l) => (
          <label key={l.id} className="flex cursor-pointer items-start gap-2.5">
            <input
              type="radio"
              name={`access-${connector.config.id}`}
              checked={current.id === l.id}
              onChange={() => void update(connector.config.id, { oauth: { scope: l.scope } })}
              className="mt-1 accent-[var(--color-primary)]"
            />
            <span>
              <span className="block text-sm text-text">{l.label}</span>
              <span className="block text-xs text-text-muted">{l.description}</span>
            </span>
          </label>
        ))}
        {connector.status.signedIn && (
          <p className="text-xs text-text-subtle">Sign in again for a change of access to apply.</p>
        )}
      </div>
    </Row>
  );
};

const ToolsDialog: React.FC<{ connector: ConnectorEntry; open: boolean; onClose: () => void }> = ({
  connector,
  open,
  onClose
}) => {
  const tools = useConnectorsStore((s) => s.tools[connector.config.id]) ?? [];
  const [q, setQ] = useState("");
  const shown = tools.filter((t) =>
    `${t.name} ${t.title ?? ""}`.toLowerCase().includes(q.toLowerCase())
  );
  return (
    <Dialog
      isOpen={open}
      title={`Tools for ${connector.config.name}`}
      onclose={onClose}
      actions={
        <Button variant="primary" size="sm" onClick={onClose}>
          Done
        </Button>
      }
    >
      <Input
        aria-label="Search tools"
        placeholder="Search tools"
        value={q}
        onChange={(e) => setQ(e.target.value)}
      />
      <ul className="custom-scrollbar mt-3 grid max-h-72 grid-cols-2 gap-x-6 gap-y-1.5 overflow-y-auto text-sm text-text">
        {shown.map((t) => (
          <li key={t.name} className="truncate" title={t.description}>
            {t.title ?? humanizeToolName(t.name)}
          </li>
        ))}
      </ul>
    </Dialog>
  );
};

/** One connector, like an app page: what it is, how to connect, what it can do, and what Dara may do with each tool. */
export const ConnectorDetail: React.FC<{ itemKey: string; onBack: () => void }> = ({
  itemKey,
  onBack
}) => {
  const entries = useConnectorsStore((s) => s.entries);
  const item: ConnectorItem | undefined = buildItems(entries).find((i) => i.key === itemKey);
  const connector = item?.connector;
  const catalog = item?.catalog;
  const id = connector?.config.id ?? "";
  const tools = useConnectorsStore((s) => s.tools[id]);
  const busy = useConnectorsStore((s) => s.busy[id] === true);
  const signingIn = useConnectorsStore((s) => s.signingIn[id] === true);
  const connect = useConnectorsStore((s) => s.connect);
  const disconnect = useConnectorsStore((s) => s.disconnect);
  const signIn = useConnectorsStore((s) => s.signIn);
  const updateConnector = useConnectorsStore((s) => s.update);
  const cancelSignIn = useConnectorsStore((s) => s.cancelSignIn);
  const signOut = useConnectorsStore((s) => s.signOut);
  const setEnabled = useConnectorsStore((s) => s.setEnabled);
  const remove = useConnectorsStore((s) => s.remove);
  const loadTools = useConnectorsStore((s) => s.loadTools);
  const [tab, setTab] = useState<Tab>("overview");
  const [menu, setMenu] = useState(false);
  const [confirmRemove, setConfirmRemove] = useState(false);
  const [showTools, setShowTools] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const ready = connector?.status.state === "ready";

  useEffect(() => {
    if (!menu) return;
    const down = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setMenu(false);
    };
    document.addEventListener("mousedown", down);
    return () => document.removeEventListener("mousedown", down);
  }, [menu]);
  useEffect(() => {
    if (ready && id) void loadTools(id);
  }, [ready, id, loadTools]);

  if (!item)
    return (
      <div className="text-left">
        <button
          type="button"
          onClick={onBack}
          className="cursor-pointer text-sm text-text-muted hover:text-text"
        >
          ← Connectors
        </button>
        <p className="mt-6 text-sm text-text-muted">That connector is not here any more.</p>
      </div>
    );

  const url = connector?.config.url ?? catalog?.url ?? null;
  const host = hostOf(url);
  const status = connector?.status;
  const needsClientId = catalog?.registers === "own-client";
  // with Dara's own client in the build, a freshly added connector already has it; one added before that still needs it saved
  const clientMissing =
    needsClientId && !connector?.config.oauth.clientId && !(catalog && builtinClient(catalog));
  // a stateless server (Google's) lists its tools without a token, so "connected" can still mean "not signed in"
  const wantsSignIn =
    !!connector && (status?.state === "needs_auth" || (!!catalog && !status?.signedIn));

  const addIt = async () => {
    if (!catalog) return;
    const problem = await addFromCatalog(catalog);
    if (problem) toast.show(problem, "error");
  };

  /** Sign in; first give the connector Dara's own client when this build has one and the connector was added without it. */
  const signInNow = async () => {
    const b = catalog ? builtinClient(catalog) : null;
    if (b && connector && !connector.config.oauth.clientId)
      await updateConnector(id, {
        oauth: { clientId: b.clientId, ...(b.clientSecret ? { clientSecret: b.clientSecret } : {}) }
      });
    void signIn(id);
  };

  const primary = (() => {
    if (!connector || !status) return null;
    if (signingIn)
      return (
        <Button variant="secondary" size="sm" onClick={() => void cancelSignIn(id)}>
          Cancel sign-in
        </Button>
      );
    if (wantsSignIn)
      return (
        <Button
          variant="primary"
          size="sm"
          disabled={clientMissing || !connector.config.enabled}
          onClick={() => void signInNow()}
        >
          {status.neededScope ? "Approve more access" : "Sign in"}
        </Button>
      );
    if (status.state === "ready" || status.state === "connecting")
      return (
        <Button variant="secondary" size="sm" disabled={busy} onClick={() => void disconnect(id)}>
          Disconnect
        </Button>
      );
    return (
      <Button
        variant="primary"
        size="sm"
        disabled={busy || !connector.config.enabled}
        onClick={() => void connect(id)}
      >
        Connect
      </Button>
    );
  })();

  return (
    <div className="text-left">
      <button
        type="button"
        onClick={onBack}
        className="flex cursor-pointer items-center gap-1.5 text-sm text-text-muted hover:text-text"
      >
        <Icon name="arrow-left" size={14} />
        Connectors
      </button>

      <div className="mt-5 flex items-start gap-4">
        <ConnectorAvatar name={item.name} size="lg" />
        <div className="min-w-0 flex-1">
          <h1 className="font-[family-name:var(--font-reply)] text-3xl font-semibold tracking-tight text-text">
            {item.name}
          </h1>
          <p className="mt-1 text-sm text-text-muted">
            By {item.publisher} · {item.tagline}
          </p>
        </div>
        {!connector && !item.soon && catalog && (
          <Button variant="primary" size="md" onClick={() => void addIt()}>
            Add
          </Button>
        )}
        {item.soon && (
          <span className="rounded-full border border-border px-2.5 py-1 text-xs text-text-subtle">
            Not available yet
          </span>
        )}
        {connector && (
          <div className="flex items-center gap-2">
            <span className="flex items-center gap-1.5 rounded-lg bg-surface-raised px-3 py-1.5 text-xs font-medium text-text-muted">
              <Icon name="check" size={13} />
              Added
            </span>
            <div ref={menuRef} className="relative">
              <button
                type="button"
                aria-label="More actions"
                aria-haspopup="menu"
                aria-expanded={menu}
                onClick={() => setMenu(!menu)}
                className="flex size-8 cursor-pointer items-center justify-center rounded-md text-text-muted hover:bg-surface-raised hover:text-text"
              >
                <Icon name="ellipsis" size={16} />
              </button>
              {menu && (
                <div
                  role="menu"
                  className="absolute right-0 top-9 z-30 w-44 rounded-lg border border-border bg-surface-raised p-1 shadow-lg"
                >
                  <button
                    type="button"
                    role="menuitem"
                    onClick={() => {
                      setMenu(false);
                      void setEnabled(id, !connector.config.enabled);
                    }}
                    className="flex w-full cursor-pointer rounded-md px-2.5 py-1.5 text-left text-xs text-text hover:bg-primary-subtle-hover"
                  >
                    {connector.config.enabled ? "Turn off" : "Turn on"}
                  </button>
                  {status?.signedIn && (
                    <button
                      type="button"
                      role="menuitem"
                      onClick={() => {
                        setMenu(false);
                        void signOut(id);
                      }}
                      className="flex w-full cursor-pointer rounded-md px-2.5 py-1.5 text-left text-xs text-text hover:bg-primary-subtle-hover"
                    >
                      Sign out
                    </button>
                  )}
                  <button
                    type="button"
                    role="menuitem"
                    onClick={() => {
                      setMenu(false);
                      setConfirmRemove(true);
                    }}
                    className="flex w-full cursor-pointer rounded-md px-2.5 py-1.5 text-left text-xs text-danger hover:bg-danger-subtle"
                  >
                    Remove
                  </button>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {connector && (
        <div
          role="tablist"
          aria-label="Connector sections"
          className="mt-6 flex gap-1 border-b border-border"
        >
          {(["overview", "permissions"] as const).map((t) => (
            <button
              key={t}
              type="button"
              role="tab"
              aria-selected={tab === t}
              onClick={() => setTab(t)}
              className={`-mb-px cursor-pointer border-b-2 px-3 py-2 text-sm font-medium transition-colors ${
                tab === t
                  ? "border-text text-text"
                  : "border-transparent text-text-muted hover:text-text"
              }`}
            >
              {t === "overview" ? "Overview" : "Permissions"}
            </button>
          ))}
        </div>
      )}

      {tab === "permissions" && connector ? (
        <PermissionsTab connectorId={id} connected={ready} />
      ) : (
        <div className="mt-6 grid gap-10 lg:grid-cols-[minmax(0,1fr)_16rem]">
          <div>
            {catalog && (
              <p className="max-w-2xl text-sm leading-relaxed text-text">{catalog.about}</p>
            )}

            {connector && status && (
              <section className="mt-7" aria-label="Connection">
                <h2 className="mb-2 text-sm font-semibold text-text">Connectors</h2>
                <div className="rounded-xl border border-border bg-surface px-4 py-3">
                  <div className="flex items-center gap-3 py-1">
                    <ConnectorAvatar name={item.name} />
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-semibold text-text">
                        {connector.config.name}
                      </p>
                      <p className="text-xs text-text-muted" role="status">
                        {signingIn
                          ? "Finish signing in in your browser. Dara is waiting."
                          : statusText(connector)}
                      </p>
                    </div>
                    {primary}
                  </div>

                  {status.state === "error" && status.error && (
                    <p
                      role="alert"
                      className="mt-2 rounded-lg border border-danger/40 bg-danger-subtle px-3 py-2 text-xs text-danger"
                    >
                      {status.error}
                    </p>
                  )}
                  {status.neededScope && (
                    <p className="mt-2 rounded-lg border border-border bg-surface-raised px-3 py-2 text-xs text-text-muted">
                      A tool needs more access than you gave ({status.neededScope}). Nothing was
                      changed; Approve more access opens your browser so you can decide.
                    </p>
                  )}

                  <dl className="mt-2 divide-y divide-border border-t border-border">
                    {url && (
                      <Row label="Server address">
                        <span className="break-all font-mono text-xs">{url}</span>{" "}
                        <button
                          type="button"
                          aria-label="Copy the server address"
                          title="Copy"
                          onClick={() => void navigator.clipboard?.writeText(url).catch(() => {})}
                          className="cursor-pointer align-middle text-text-subtle hover:text-text"
                        >
                          <Icon name="copy" size={13} />
                        </button>
                      </Row>
                    )}
                    <Row label="Sign-in">
                      {catalog
                        ? "Required"
                        : connector.config.kind === "http"
                          ? "If the server asks"
                          : "Not used"}
                    </Row>
                    {catalog && needsClientId && (
                      <ClientForm entry={catalog} connector={connector} />
                    )}
                    {catalog && <AccessLevel entry={catalog} connector={connector} />}
                    <Row label="Tools">
                      {ready && tools === undefined ? (
                        <Skeleton className="h-4 w-48" />
                      ) : tools && tools.length ? (
                        <>
                          <span className="mr-2">{tools.length}</span>
                          <span className="text-xs text-text-muted">
                            {tools
                              .slice(0, 5)
                              .map((t) => t.name)
                              .join(", ")}
                            {tools.length > 5 && ", ..."}
                          </span>
                          <button
                            type="button"
                            onClick={() => setShowTools(true)}
                            className="mt-1 block cursor-pointer text-xs text-text-muted underline underline-offset-2 hover:text-text"
                          >
                            Show all {tools.length}
                          </button>
                        </>
                      ) : (
                        <span className="text-text-muted">Connect to see the tools.</span>
                      )}
                    </Row>
                  </dl>
                </div>
              </section>
            )}
          </div>

          <aside className="space-y-5">
            <Side label="Developer">{item.publisher}</Side>
            {host && (
              <Side label="Safety and access">
                Can send data to <span className="break-all">{host}</span>
              </Side>
            )}
            {catalog && <Side label="Authorization">Sign-in required</Side>}
            {catalog && catalog.categories.length > 0 && (
              <Side label="Categories">
                <div className="flex flex-wrap gap-1.5">
                  {catalog.categories.map((c) => (
                    <span
                      key={c}
                      className="rounded-lg bg-surface-raised px-2.5 py-1 text-xs text-text-muted"
                    >
                      {c}
                    </span>
                  ))}
                </div>
              </Side>
            )}
            {connector && <Side label="Added">{relativeTime(connector.config.createdAt)}</Side>}
          </aside>
        </div>
      )}

      {connector && (
        <ToolsDialog connector={connector} open={showTools} onClose={() => setShowTools(false)} />
      )}
      <Dialog
        isOpen={confirmRemove}
        title={`Remove ${item.name}?`}
        description="Its saved permissions, call history, sign-in and stored variables are deleted from this computer."
        onclose={() => setConfirmRemove(false)}
        actions={
          <>
            <Button variant="ghost" size="sm" onClick={() => setConfirmRemove(false)}>
              Cancel
            </Button>
            <Button
              variant="danger"
              size="sm"
              onClick={() => {
                setConfirmRemove(false);
                void remove(id).then(onBack);
              }}
            >
              Remove
            </Button>
          </>
        }
      />
    </div>
  );
};
