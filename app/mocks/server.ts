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

export function createMockServer(options: MockServerOptions = {}): {
  server: http.Server;
  start: () => Promise<number>;
  stop: () => Promise<void>;
} {
  const server = http.createServer((req, res) => {
    // Enable CORS for testing & local webview dev
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
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

    if (pathname === "/v1/chat/completions" && req.method === "POST") {
      let body = "";
      req.on("data", (chunk) => {
        body += chunk;
      });
      req.on("end", () => {
        let isStream = false;
        try {
          const parsed = JSON.parse(body);
          isStream = Boolean(parsed.stream);
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
