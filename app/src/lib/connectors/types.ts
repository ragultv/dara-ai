// Connectors (docs/WORKFLOWS.md): programs or servers that speak MCP (the Model Context Protocol) and give Dara tools, such as reading a calendar.
// The connection, the permissions and the audit log live in the desktop app's Rust layer; these are the shapes the UI gets from it.

export type ConnectorKind = "stdio" | "http";
/** What the user decided for one tool. Only the user ever sets this; a server's own hints never grant anything. */
export type ToolPermission = "ask" | "allow" | "deny";
export type ConnectionState = "disconnected" | "connecting" | "ready" | "needs_auth" | "error";

export interface ConnectorConfig {
  id: string;
  name: string;
  kind: ConnectorKind;
  /** stdio: the program to start (no shell) */
  command: string | null;
  args: string[];
  /** the NAMES of the environment variables given to a local program; their values are kept in the system credential store */
  envNames: string[];
  /** http: the server address (https, or this computer) */
  url: string | null;
  enabled: boolean;
  createdAt: number;
}

export interface ConnectorStatus {
  state: ConnectionState;
  /** why it is in the error state */
  message: string | null;
  serverName: string | null;
  serverVersion: string | null;
  protocol: string | null;
  toolCount: number;
}

export interface ConnectorEntry {
  config: ConnectorConfig;
  status: ConnectorStatus;
}

export interface ConnectorTool {
  name: string;
  title: string | null;
  description: string;
  inputSchema: Record<string, unknown>;
  /** the server's own claim; shown, never trusted */
  readOnlyHint: boolean | null;
  permission: ToolPermission;
  /** the tool's definition changed since the user last chose a permission (so it was put back to "ask") */
  changed: boolean;
  /** false when the server no longer offers it */
  available: boolean;
}

export type CallOutcome = "ok" | "error" | "needs_approval" | "denied";

export interface ToolCallResult {
  outcome: CallOutcome;
  /** MCP content blocks as the server sent them (text, image, resource...) */
  content: Array<Record<string, unknown>>;
  structuredContent: unknown;
  isError: boolean;
  truncated: boolean;
}

export interface AuditEntry {
  id: number;
  connectorId: string;
  tool: string;
  argsJson: string;
  outcome: CallOutcome;
  durationMs: number;
  resultChars: number;
  createdAt: number;
}

export interface AddConnectorInput {
  name: string;
  kind: ConnectorKind;
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  url?: string;
}

export interface ConnectorPatch {
  name?: string;
  enabled?: boolean;
  args?: string[];
  env?: Record<string, string>;
}

/** What the UI needs from the desktop layer. The desktop app implements it with Tauri commands; a browser tab gets a demo with sample data. */
export interface ConnectorsBackend {
  readonly isDemo: boolean;
  list(): Promise<ConnectorEntry[]>;
  add(input: AddConnectorInput): Promise<ConnectorConfig>;
  update(id: string, patch: ConnectorPatch): Promise<void>;
  remove(id: string): Promise<void>;
  connect(id: string): Promise<ConnectorStatus>;
  disconnect(id: string): Promise<void>;
  tools(id: string): Promise<ConnectorTool[]>;
  setPermission(id: string, tool: string, permission: ToolPermission): Promise<void>;
  call(
    id: string,
    tool: string,
    args: Record<string, unknown>,
    approved: boolean
  ): Promise<ToolCallResult>;
  audit(id: string | null, limit: number): Promise<AuditEntry[]>;
  stderr(id: string): Promise<string>;
}
