import React, { useEffect, useState } from "react";
import { Dialog } from "../ui/Dialog.tsx";
import { Button } from "../ui/Button.tsx";
import { Textarea } from "../ui/Textarea.tsx";
import { useConnectorsStore } from "../stores/useConnectorsStore.ts";
import { skeletonArgs } from "./validate.ts";
import type { ConnectorTool, ToolCallResult } from "./types.ts";

/** The text of a result. Anything a server sends is untrusted, so it is shown as plain text, never as HTML or Markdown. */
export function resultText(r: ToolCallResult): string {
  const parts = r.content.map((c) => {
    if (c["type"] === "text" && typeof c["text"] === "string") return c["text"];
    if (c["type"] === "image") return "[an image]";
    if (c["type"] === "resource_link" && typeof c["uri"] === "string") return `[link] ${c["uri"]}`;
    return `[${String(c["type"] ?? "content")}]`;
  });
  if (!parts.length && r.structuredContent !== null && r.structuredContent !== undefined)
    parts.push(JSON.stringify(r.structuredContent, null, 2));
  return parts.join("\n");
}

/** Try one tool by hand: the arguments, then what came back. A tool set to Ask runs only after you confirm. */
export const ToolTestDialog: React.FC<{
  connectorId: string;
  tool: ConnectorTool | null;
  onClose: () => void;
}> = ({ connectorId, tool, onClose }) => {
  const call = useConnectorsStore((s) => s.call);
  const [text, setText] = useState("{}");
  const [result, setResult] = useState<ToolCallResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    if (tool) {
      setText(JSON.stringify(skeletonArgs(tool.inputSchema), null, 2));
      setResult(null);
      setError(null);
    }
  }, [tool]);

  const run = async (approved: boolean) => {
    if (!tool) return;
    let args: Record<string, unknown>;
    try {
      const parsed: unknown = JSON.parse(text || "{}");
      if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed))
        throw new Error("not an object");
      args = parsed as Record<string, unknown>;
    } catch {
      setError('The arguments must be a JSON object, for example {"range": "today"}.');
      return;
    }
    setError(null);
    setRunning(true);
    const r = await call(connectorId, tool.name, args, approved);
    setRunning(false);
    if ("error" in r) setError(r.error);
    else setResult(r);
  };

  const needsApproval = result?.outcome === "needs_approval";

  return (
    <Dialog
      isOpen={tool !== null}
      title={tool ? `Try ${tool.title ?? tool.name}` : ""}
      description={tool?.description ?? ""}
      onclose={onClose}
      actions={
        <>
          <Button variant="ghost" size="sm" onClick={onClose}>
            Close
          </Button>
          {needsApproval ? (
            <Button variant="primary" size="sm" disabled={running} onClick={() => void run(true)}>
              Run it once
            </Button>
          ) : (
            <Button variant="primary" size="sm" disabled={running} onClick={() => void run(false)}>
              Run
            </Button>
          )}
        </>
      }
    >
      <div className="space-y-3 text-left">
        <Textarea
          label="Arguments (JSON)"
          rows={5}
          value={text}
          onChange={(e) => setText(e.target.value)}
          spellCheck={false}
          className="font-mono text-xs"
        />
        {error && (
          <p role="alert" className="text-xs text-danger">
            {error}
          </p>
        )}
        {needsApproval && (
          <p className="text-xs text-text-muted">
            This tool is set to Ask, so nothing ran. Run it once to see what it does, or change its
            permission in the list.
          </p>
        )}
        {result?.outcome === "denied" && (
          <p className="text-xs text-danger">This tool is blocked, so it did not run.</p>
        )}
        {result && (result.outcome === "ok" || result.outcome === "error") && (
          <div>
            <p className="mb-1 text-xs font-semibold text-text">
              {result.outcome === "ok" ? "Result" : "The tool reported an error"}
              {result.truncated ? " (cut short)" : ""}
            </p>
            <pre
              data-testid="tool-result"
              className="custom-scrollbar max-h-64 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-border bg-surface-sunken p-3 text-xs text-text"
            >
              {resultText(result) || "(nothing)"}
            </pre>
          </div>
        )}
      </div>
    </Dialog>
  );
};
