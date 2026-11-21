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
  const [trust, setTrust] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const close = () => {
    setName("");
    setCommand("");
    setArgs("");
    setEnv("");
    setUrl("");
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
        : { name, kind, url };
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
            hint="An https address such as https://example.com/mcp. Signing in to the server comes in the next version."
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            spellCheck={false}
          />
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
