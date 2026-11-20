import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const repoRoot = path.resolve(__dirname, "..", "..");
const samplesDir = path.resolve(repoRoot, "docs", "ui", "samples");

export interface MockServerOptions {
  port?: number;
  apiKey?: string;
}

interface MockArtifact {
  id: string;
  kind: string;
  title: string;
  content: string;
  pinned: boolean;
  created_at: number;
  updated_at: number;
}

export function createMockServer(options: MockServerOptions = {}): {
  server: http.Server;
  start: () => Promise<number>;
  stop: () => Promise<void>;
} {
  const artifacts = new Map<string, MockArtifact>();
  // version history (docs/ui/09-artifacts.md): the mock keeps a version only for an explicit `snapshot` label or a restore
  const versions = new Map<
    string,
    Array<{ rev: number; title: string; content: string; label: string; created_at: number }>
  >();
  const keepVersion = (a: MockArtifact, label: string) => {
    if (!a.title && !a.content) return;
    const list = versions.get(a.id) ?? [];
    const last = list[list.length - 1];
    if (last && last.title === a.title && last.content === a.content) return;
    list.push({
      rev: (last?.rev ?? 0) + 1,
      title: a.title,
      content: a.content,
      label,
      created_at: a.updated_at
    });
    versions.set(a.id, list);
  };
  const server = http.createServer((req, res) => {
    // Enable CORS for testing & local webview dev
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization, Accept");

    if (req.method === "OPTIONS") {
      res.writeHead(204);
      res.end();
      return;
    }

    if (options.apiKey) {
      const authHeader = req.headers.authorization;
      if (!authHeader || authHeader !== `Bearer ${options.apiKey}`) {
        res.writeHead(401, { "Content-Type": "application/json" });
        res.end(
          JSON.stringify({
            error: {
              message: "Unauthorized",
              type: "authentication_error"
            }
          })
        );
        return;
      }
    }

    const url = new URL(req.url ?? "/", `http://${req.headers.host ?? "localhost"}`);
    const pathname = url.pathname;

    if (pathname === "/health") {
      const healthPath = path.join(samplesDir, "health.json");
      const data = fs.readFileSync(healthPath, "utf-8");
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(data);
      return;
    }

    if (pathname === "/v1/models" || pathname === "/dara/models") {
      const modelsPath = path.join(samplesDir, "models.json");
      const data = fs.readFileSync(modelsPath, "utf-8");
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(data);
      return;
    }

    if (pathname.startsWith("/dara/models/") && pathname.endsWith("/options")) {
      const optionsPath = path.join(samplesDir, "options-nano.json");
      const data = fs.readFileSync(optionsPath, "utf-8");
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(data);
      return;
    }

    if (pathname.includes("/benchmark") && req.method === "POST") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          id: "job-bm-1",
          status: "running",
          log: ["Initializing benchmark on GPU...", "Measuring INT8 prefill and decode speeds..."]
        })
      );
      return;
    }

    if (pathname.startsWith("/dara/benchmarks/") && req.method === "GET") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          id: "job-bm-1",
          status: "done",
          log: [
            "Benchmark completed in 32s.",
            "Measured decode: 26.5 tok/s (empty context)",
            "Estimated decode: 22.8 tok/s (full context)"
          ]
        })
      );
      return;
    }

    if (pathname.startsWith("/dara/benchmarks/") && req.method === "DELETE") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          id: pathname.split("/").pop() ?? "job-bm-1",
          status: "cancelled",
          log: ["Benchmark cancelled by user."]
        })
      );
      return;
    }

    // App database endpoints (docs/ui/08-database.md): reads serve the recorded samples, writes are accepted and echo a plausible reply.
    const sampleFor: Record<string, string> = {
      "/dara/store": "store-info.json",
      "/dara/settings": "settings.json",
      "/dara/conversations": "conversations.json",
      "/dara/model-configs": "model-configs.json",
      "/dara/search": "search.json"
    };
    const sampleFile = sampleFor[pathname];
    if (sampleFile && req.method === "GET") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(fs.readFileSync(path.join(samplesDir, sampleFile), "utf-8"));
      return;
    }
    if (
      pathname.startsWith("/dara/conversations/") &&
      req.method === "GET" &&
      !pathname.endsWith("/messages")
    ) {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(fs.readFileSync(path.join(samplesDir, "conversation.json"), "utf-8"));
      return;
    }
    if (
      (pathname.startsWith("/dara/conversations") ||
        pathname.startsWith("/dara/messages") ||
        pathname.startsWith("/dara/settings") ||
        pathname.startsWith("/dara/model-configs") ||
        pathname === "/dara/data") &&
      req.method !== "GET"
    ) {
      res.writeHead(req.method === "POST" ? 201 : 200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ deleted: req.method === "DELETE" }));
      return;
    }

    if (pathname === "/dara/stats") {
      const statsPath = path.join(samplesDir, "stats.json");
      const data = fs.readFileSync(statsPath, "utf-8");
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(data);
      return;
    }

    if (pathname === "/dara/version") {
      const versionPath = path.join(samplesDir, "version.json");
      const data = fs.readFileSync(versionPath, "utf-8");
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(data);
      return;
    }

    if (pathname === "/dara/load") {
      const loadPath = path.join(samplesDir, "load-status.json");
      const data = fs.readFileSync(loadPath, "utf-8");
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(data);
      return;
    }

    if (pathname === "/dara/models/pull" && req.method === "POST") {
      let body = "";
      req.on("data", (chunk) => {
        body += chunk;
      });
      req.on("end", () => {
        let model = "dara-nano";
        try {
          const parsed = JSON.parse(body);
          if (parsed.model) model = parsed.model;
        } catch {
          // ignore
        }
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(
          JSON.stringify({
            id: "job-pull-123",
            model,
            status: "downloading",
            file: "model",
            files_done: 0,
            files_total: 1,
            completed_bytes: 540000000,
            total_bytes: 800000000,
            percent: 67.5,
            mb_per_second: 34.2,
            eta_seconds: 7,
            seconds: 14
          })
        );
      });
      return;
    }

    if (pathname.startsWith("/dara/models/pull/") && req.method === "GET") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          id: "job-pull-123",
          model: "dara-nano",
          status: "done",
          file: "model",
          files_done: 1,
          files_total: 1,
          completed_bytes: 800000000,
          total_bytes: 800000000,
          percent: 100,
          mb_per_second: 35.0,
          eta_seconds: 0,
          seconds: 22
        })
      );
      return;
    }

    if (pathname.startsWith("/dara/models/pull/") && req.method === "DELETE") {
      res.writeHead(204);
      res.end();
      return;
    }

    if (pathname === "/dara/models/load" && req.method === "POST") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          state: "ready",
          stage: "ready",
          percent: 100,
          model: "dara-nano",
          seconds: 2.1
        })
      );
      return;
    }

    if (pathname === "/dara/models/unload" && req.method === "POST") {
      res.writeHead(204);
      res.end();
      return;
    }

    if (pathname.startsWith("/dara/models/") && req.method === "DELETE") {
      res.writeHead(204);
      res.end();
      return;
    }

    // ---- artifacts (docs/ui/09-artifacts.md): kept in memory for the dev mock ----
    const artifactMatch =
      /^\/dara\/artifacts(?:\/([0-9a-f-]{36})(?:\/(versions|restore)(?:\/(\d+))?)?)?$/i.exec(
        pathname
      );
    if (artifactMatch) {
      const id = artifactMatch[1];
      const sub = artifactMatch[2];
      const subRev = artifactMatch[3] ? Number(artifactMatch[3]) : undefined;
      const send = (status: number, body: unknown) => {
        res.writeHead(status, { "Content-Type": "application/json" });
        res.end(JSON.stringify(body));
      };
      let raw = "";
      req.on("data", (c) => {
        raw += c;
      });
      req.on("end", () => {
        const body = raw ? JSON.parse(raw) : {};
        const full = (a: MockArtifact) => ({ ...a, chars: a.content.length });
        const summary = (a: MockArtifact) => {
          const { content, ...rest } = a;
          return { ...rest, chars: content.length, snippet: content.slice(0, 600) };
        };
        if (!id && req.method === "GET") {
          const kind = url.searchParams.get("kind");
          const data = [...artifacts.values()]
            .filter((a) => !kind || a.kind === kind)
            .sort((a, b) => b.updated_at - a.updated_at);
          return send(200, {
            object: "list",
            data: data.map(summary),
            total: data.length,
            limit: 200,
            offset: 0
          });
        }
        if (!id && req.method === "POST") {
          const now = Date.now();
          const a: MockArtifact = {
            id: body.id ?? crypto.randomUUID(),
            kind: body.kind ?? "doc",
            title: body.title ?? "",
            content: body.content ?? "",
            pinned: false,
            created_at: now,
            updated_at: now
          };
          artifacts.set(a.id, a);
          return send(201, full(a));
        }
        const a = id ? artifacts.get(id) : undefined;
        if (!a) return send(404, { error: { message: "artifact not found", type: "not_found" } });
        if (sub === "versions" && req.method === "GET") {
          const list = [...(versions.get(a.id) ?? [])].reverse();
          if (subRev !== undefined) {
            const v = list.find((x) => x.rev === subRev);
            if (!v)
              return send(404, { error: { message: "version not found", type: "not_found" } });
            return send(200, { ...v, chars: v.content.length });
          }
          return send(200, {
            object: "list",
            data: list.map(({ content, ...rest }) => ({
              ...rest,
              chars: content.length,
              snippet: content.slice(0, 200)
            })),
            total: list.length,
            limit: 100,
            offset: 0
          });
        }
        if (sub === "restore" && req.method === "POST") {
          const v = (versions.get(a.id) ?? []).find((x) => x.rev === body.rev);
          if (!v) return send(404, { error: { message: "version not found", type: "not_found" } });
          keepVersion(a, "Before restore");
          a.title = v.title;
          a.content = v.content;
          a.updated_at = Date.now();
          return send(200, full(a));
        }
        if (req.method === "GET") return send(200, full(a));
        if (req.method === "PATCH") {
          if (typeof body.snapshot === "string") keepVersion(a, body.snapshot);
          if (typeof body.title === "string") a.title = body.title;
          if (typeof body.content === "string") a.content = body.content;
          if (typeof body.pinned === "boolean") a.pinned = body.pinned;
          a.updated_at = Date.now();
          return send(200, full(a));
        }
        if (req.method === "DELETE") {
          artifacts.delete(a.id);
          return send(200, { deleted: true });
        }
        return send(405, {
          error: { message: "method not allowed", type: "invalid_request_error" }
        });
      });
      return;
    }

    if (pathname === "/v1/chat/completions" && req.method === "POST") {
      let body = "";
      req.on("data", (chunk) => {
        body += chunk;
      });
      req.on("end", () => {
        let isStream = false;
        let thinking = false;
        try {
          const parsed = JSON.parse(body);
          isStream = Boolean(parsed.stream);
          thinking =
            typeof parsed.reasoning_effort === "string" && parsed.reasoning_effort !== "none";
        } catch {
          // default false
        }

        if (isStream) {
          const streamPath = path.join(samplesDir, "chat-stream.sse");
          const data = fs.readFileSync(streamPath, "utf-8");
          res.writeHead(200, {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            Connection: "keep-alive"
          });
          if (thinking) {
            // reasoning deltas first, then the recorded answer (the shape the engine streams)
            const thoughts = [
              "Let me work through this. ",
              "First, restate the question. ",
              "Then check the answer once more."
            ];
            const mk = (delta: Record<string, string>) =>
              `data: ${JSON.stringify({ id: "chatcmpl-mock", object: "chat.completion.chunk", created: 0, model: "dara-nano", choices: [{ index: 0, finish_reason: null, delta }] })}\n\n`;
            res.write(mk({ role: "assistant" } as unknown as Record<string, string>));
            for (const t of thoughts) res.write(mk({ reasoning_content: t }));
          }
          res.end(data);
        } else {
          const chatPath = path.join(samplesDir, "chat-response.json");
          const data = fs.readFileSync(chatPath, "utf-8");
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(data);
        }
      });
      return;
    }

    res.writeHead(404, { "Content-Type": "application/json" });
    res.end(
      JSON.stringify({
        error: {
          message: `Endpoint ${pathname} not found in mock server`,
          type: "not_found"
        }
      })
    );
  });

  return {
    server,
    start: () =>
      new Promise<number>((resolve) => {
        const port = options.port ?? 0;
        server.listen(port, "127.0.0.1", () => {
          const address = server.address();
          if (address && typeof address === "object") {
            resolve(address.port);
          } else {
            resolve(port);
          }
        });
      }),
    stop: () =>
      new Promise<void>((resolve, reject) => {
        server.close((err) => {
          if (err) reject(err);
          else resolve();
        });
      })
  };
}

// Standalone execution support
if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(__filename)) {
  const portArg = process.argv[2] ? parseInt(process.argv[2], 10) : 11435;
  const mock = createMockServer({ port: portArg });
  mock.start().then((port) => {
    console.log(`Dara mock server running on http://127.0.0.1:${port}`);
    console.log(`Serving samples from ${samplesDir}`);
  });
}
