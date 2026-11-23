import React, { useState } from "react";
import { Dialog } from "../ui/Dialog.tsx";
import { Button } from "../ui/Button.tsx";
import { Input } from "../ui/Input.tsx";
import { Textarea } from "../ui/Textarea.tsx";
import { Checkbox } from "../ui/Checkbox.tsx";
import { SegmentedControl } from "../ui/SegmentedControl.tsx";
import { useConnectorsStore } from "../stores/useConnectorsStore.ts";
import { parseArgs, parseEnv } from "./validate.ts";
import type { AddConnectorInput, ConnectorKind } from "./types.ts";

/** Adding a connector: a program on this computer (stdio) or a server address (https). */
export const AddConnectorDialog: React.FC<{ isOpen: boolean; onClose: () => void }> = ({
  isOpen,
  onClose
}) => {
  const add = useConnectorsStore((s) => s.add);
  const [kind, setKind] = useState<ConnectorKind>("stdio");
  const [name, setName] = useState("");
  const [command, setCommand] = useState("");
  const [args, setArgs] = useState("");
  const [env, setEnv] = useState("");
  const [url, setUrl] = useState("");
  const [clientId, setClientId] = useState("");
  const [clientSecret, setClientSecret] = useState("");
  const [scope, setScope] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [keyHeader, setKeyHeader] = useState("");
  const [trust, setTrust] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const close = () => {
    setName("");
    setCommand("");
    setArgs("");
    setEnv("");
    setUrl("");
    setClientId("");
    setClientSecret("");
    setScope("");
    setApiKey("");
    setKeyHeader("");
    setTrust(false);
    setError(null);
    onClose();
  };

  const submit = async () => {
    const parsedEnv = parseEnv(env);
    if (kind === "stdio" && parsedEnv.bad) {
      setError(`"${parsedEnv.bad}" is not a variable. Write NAME=value on each line.`);
      return;
    }
    const input: AddConnectorInput =
      kind === "stdio"
        ? { name, kind, command, args: parseArgs(args), env: parsedEnv.env }
        : {
            name,
            kind,
            url,
            ...(clientId.trim() || clientSecret || scope.trim()
              ? {
                  oauth: {
                    ...(clientId.trim() ? { clientId: clientId.trim() } : {}),
                    ...(clientSecret ? { clientSecret } : {}),
                    ...(scope.trim() ? { scope: scope.trim() } : {})
                  }
                }
              : {}),
            ...(apiKey.trim() || keyHeader.trim()
              ? {
                  apiKey: {
                    value: apiKey,
                    ...(keyHeader.trim() ? { header: keyHeader.trim() } : {})
                  }
                }
              : {})
          };
    setBusy(true);
    const problem = await add(input);
    setBusy(false);
    if (problem) setError(problem);
    else close();
  };

  return (
    <Dialog
      isOpen={isOpen}
      title="Add a connector"
      description="A connector gives Dara tools, for example reading your calendar. Dara asks before using any of them."
      onclose={close}
      actions={
        <>
          <Button variant="ghost" size="sm" onClick={close}>
            Cancel
          </Button>
          <Button
            variant="primary"
            size="sm"
            disabled={busy || (kind === "stdio" && !trust)}
            onClick={() => void submit()}
          >
            Add connector
          </Button>
        </>
      }
    >
      <div className="space-y-4 text-left">
        <SegmentedControl
          label="Type"
          value={kind}
          onchange={(v) => {
            setKind(v as ConnectorKind);
            setError(null);
          }}
          options={[
            { value: "stdio", label: "Program on this computer" },
            { value: "http", label: "Server address" }
          ]}
        />
        <Input label="Name" value={name} onChange={(e) => setName(e.target.value)} maxLength={60} />
        {kind === "stdio" ? (
          <>
            <Input
              label="Program"
              hint="The program that starts the connector, for example npx or a full path."
              value={command}
              onChange={(e) => setCommand(e.target.value)}
            />
            <Textarea
              label="Arguments (one per line)"
              rows={3}
              value={args}
              onChange={(e) => setArgs(e.target.value)}
            />
            <Textarea
              label="Environment variables (NAME=value, one per line)"
              hint="Values are kept in the Windows credential store, not in Dara's database."
              rows={2}
              value={env}
              onChange={(e) => setEnv(e.target.value)}
              spellCheck={false}
            />
            <Checkbox
              checked={trust}
              onchange={setTrust}
              label="I trust this program"
              description="It runs on this computer with your permissions and can read your files, so add only programs you trust."
            />
          </>
        ) : (
          <Input
            label="Server address"
            hint="An https address such as https://example.com/mcp. If the server asks you to sign in, Dara opens your browser when you press Sign in."
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            spellCheck={false}
          />
        )}
        {kind === "http" && (
          <details className="rounded-lg border border-border px-3 py-2">
            <summary className="cursor-pointer text-xs font-semibold text-text-muted">
              Sign-in settings (only if the server needs your own client ID)
            </summary>
            <div className="mt-3 space-y-3">
              <p className="text-xs text-text-muted">
                Most servers register Dara by themselves. Some, such as Google's, need a client you
                create in their console. The secret is kept in the Windows credential store.
              </p>
              <Input
                label="Client ID"
                value={clientId}
                onChange={(e) => setClientId(e.target.value)}
                spellCheck={false}
              />
              <Input
                label="Client secret (if the console gave you one)"
                type="password"
                value={clientSecret}
                onChange={(e) => setClientSecret(e.target.value)}
                autoComplete="off"
              />
              <Input
                label="Scope (optional)"
                value={scope}
                onChange={(e) => setScope(e.target.value)}
                spellCheck={false}
              />
            </div>
          </details>
        )}
        {kind === "http" && (
          <details className="rounded-lg border border-border px-3 py-2">
            <summary className="cursor-pointer text-xs font-semibold text-text-muted">
              API key (only if the server wants a key instead of a sign-in)
            </summary>
            <div className="mt-3 space-y-3">
              <p className="text-xs text-text-muted">
                Paste the key the service gave you. It is kept in the Windows credential store, never
                in Dara's database, and is sent with every request to this server.
              </p>
              <Input
                label="API key"
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                autoComplete="off"
                spellCheck={false}
              />
              <Input
                label="Header (optional)"
                hint="Left empty, the key is sent as Authorization: Bearer <key>. Some servers want it in their own header, for example X-API-Key."
                value={keyHeader}
                onChange={(e) => setKeyHeader(e.target.value)}
                spellCheck={false}
              />
            </div>
          </details>
        )}
        {error && (
          <p role="alert" className="text-xs text-danger">
            {error}
          </p>
        )}
      </div>
    </Dialog>
  );
};
