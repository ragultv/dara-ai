import { ApiError } from "./errors.ts";
import { parseSseStream } from "./sse.ts";
import type {
  Health,
  ModelList,
  DownloadJob,
  PullRequest,
  LoadRequest,
  LoadStatus,
  Stats,
  VersionInfo,
  ChatRequest,
  ChatChunk,
  ChatResponse,
  OptionsResponse,
  BenchmarkJob
} from "./types.ts";

export interface ClientConfig {
  baseUrl: string;
  apiKey?: string | undefined;
  defaultTimeoutMs?: number | undefined;
}

export class DaraApiClient {
  private baseUrl: string;
  private apiKey?: string | undefined;
  private defaultTimeoutMs: number;
  private authRefresher?: (() => Promise<boolean>) | undefined;

  constructor(config: ClientConfig = { baseUrl: "http://127.0.0.1:11435" }) {
    this.baseUrl = config.baseUrl.replace(/\/+$/, "");
    this.apiKey = config.apiKey;
    this.defaultTimeoutMs = config.defaultTimeoutMs ?? 30000;
  }

  setConnection(baseUrl: string, apiKey?: string | undefined): void {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.apiKey = apiKey;
  }

  /**
   * A function that fetches the engine's current key (the desktop app generates a new one each time it restarts the engine).
   * When a request is answered 401 it is called once; if it returns true the request is repeated with the new key.
   */
  setAuthRefresher(fn: (() => Promise<boolean>) | undefined): void {
    this.authRefresher = fn;
  }

  getBaseUrl(): string {
    return this.baseUrl;
  }

  async request<T>(
    endpoint: string,
    options: RequestInit & { timeoutMs?: number | undefined; retriedAuth?: boolean } = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint.startsWith("/") ? endpoint : `/${endpoint}`}`;
    const timeoutMs = options.timeoutMs ?? this.defaultTimeoutMs;
    const controller = new AbortController();

    const headers = new Headers(options.headers);
    if (!headers.has("Accept")) {
      headers.set("Accept", "application/json");
    }
    if (this.apiKey) {
      headers.set("Authorization", `Bearer ${this.apiKey}`);
    }
    if (options.body && typeof options.body === "string" && !headers.has("Content-Type")) {
      headers.set("Content-Type", "application/json");
    }

    const timer = setTimeout(() => {
      controller.abort();
    }, timeoutMs);

    if (options.signal) {
      options.signal.addEventListener("abort", () => {
        controller.abort();
      });
      if (options.signal.aborted) {
        controller.abort();
      }
    }

    const fetchOptions: RequestInit = {
      ...options,
      headers,
      signal: controller.signal
    };

    try {
      const response = await fetch(url, fetchOptions);

      if (response.status === 401 && this.authRefresher && !options.retriedAuth) {
        const refreshed = await this.authRefresher().catch(() => false);
        if (refreshed) return await this.request<T>(endpoint, { ...options, retriedAuth: true });
      }

      if (!response.ok) {
        // read the body once (json() would use it up and leave nothing for text())
        const raw = await response.text().catch(() => "");
        let errorBody: unknown = raw;
        try {
          errorBody = JSON.parse(raw);
        } catch {
          // not JSON: keep the text
        }
        throw ApiError.fromResponseBody(
          response.status,
          errorBody,
          `Request to ${endpoint} failed with status ${response.status}`
        );
      }

      if (response.status === 204) {
        return undefined as unknown as T;
      }

      return (await response.json()) as T;
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        throw err;
      }
      if (err instanceof DOMException && err.name === "AbortError") {
        throw new ApiError(408, "Request timed out or was aborted", "timeout");
      }
      const message = err instanceof Error ? err.message : "Network error";
      throw new ApiError(0, message, "network_error");
    } finally {
      clearTimeout(timer);
    }
  }

  /** GET that returns the raw bytes (images): an <img src> cannot send the Authorization header. */
  async requestBlob(endpoint: string, options: { signal?: AbortSignal } = {}): Promise<Blob> {
    const url = `${this.baseUrl}${endpoint.startsWith("/") ? endpoint : `/${endpoint}`}`;
    const headers = new Headers();
    if (this.apiKey) headers.set("Authorization", `Bearer ${this.apiKey}`);
    let response: Response;
    try {
      response = await fetch(url, {
        method: "GET",
        headers,
        ...(options.signal ? { signal: options.signal } : {})
      });
    } catch (err: unknown) {
      throw new ApiError(0, err instanceof Error ? err.message : "Network error", "network_error");
    }
    if (!response.ok) {
      let body: unknown;
      try {
        body = await response.json();
      } catch {
        body = await response.text();
      }
      throw ApiError.fromResponseBody(
        response.status,
        body,
        `Request to ${endpoint} failed with status ${response.status}`
      );
    }
    return response.blob();
  }

  async getHealth(signal?: AbortSignal, timeoutMs = 3000): Promise<Health> {
    const opts: RequestInit & { timeoutMs: number } = {
      method: "GET",
      timeoutMs
    };
    if (signal) {
      opts.signal = signal;
    }
    return this.request<Health>("/health", opts);
  }

  async getModels(signal?: AbortSignal): Promise<ModelList> {
    const opts: RequestInit = {
      method: "GET"
    };
    if (signal) {
      opts.signal = signal;
    }
    return this.request<ModelList>("/v1/models", opts);
  }

  async pullModel(req: PullRequest, signal?: AbortSignal): Promise<DownloadJob> {
    const opts: RequestInit = {
      method: "POST",
      body: JSON.stringify(req)
    };
    if (signal) {
      opts.signal = signal;
    }
    return this.request<DownloadJob>("/dara/models/pull", opts);
  }

  async getPullProgress(jobId: string, signal?: AbortSignal): Promise<DownloadJob> {
    const opts: RequestInit = {
      method: "GET"
    };
    if (signal) {
      opts.signal = signal;
    }
    return this.request<DownloadJob>(`/dara/models/pull/${jobId}`, opts);
  }

  async cancelPull(jobId: string, signal?: AbortSignal): Promise<void> {
    const opts: RequestInit = {
      method: "DELETE"
    };
    if (signal) {
      opts.signal = signal;
    }
    await this.request<undefined>(`/dara/models/pull/${jobId}`, opts);
  }

  async loadModel(req: LoadRequest, signal?: AbortSignal): Promise<LoadStatus> {
    const opts: RequestInit = {
      method: "POST",
      body: JSON.stringify({ ...req, async: true })
    };
    if (signal) {
      opts.signal = signal;
    }
    return this.request<LoadStatus>("/dara/models/load", opts);
  }

  async getLoadStatus(signal?: AbortSignal): Promise<LoadStatus> {
    const opts: RequestInit = {
      method: "GET"
    };
    if (signal) {
      opts.signal = signal;
    }
    return this.request<LoadStatus>("/dara/load", opts);
  }

  async unloadModel(signal?: AbortSignal): Promise<void> {
    const opts: RequestInit = {
      method: "POST"
    };
    if (signal) {
      opts.signal = signal;
    }
    await this.request<undefined>("/dara/models/unload", opts);
  }

  async deleteModel(modelId: string, signal?: AbortSignal): Promise<void> {
    const opts: RequestInit = {
      method: "DELETE"
    };
    if (signal) {
      opts.signal = signal;
    }
    await this.request<undefined>(`/dara/models/${modelId}`, opts);
  }

  async getStats(signal?: AbortSignal): Promise<Stats> {
    const opts: RequestInit = {
      method: "GET"
    };
    if (signal) {
      opts.signal = signal;
    }
    return this.request<Stats>("/dara/stats", opts);
  }

  async getVersion(signal?: AbortSignal): Promise<VersionInfo> {
    const opts: RequestInit = {
      method: "GET"
    };
    if (signal) {
      opts.signal = signal;
    }
    return this.request<VersionInfo>("/dara/version", opts);
  }

  async shutdown(signal?: AbortSignal): Promise<void> {
    const opts: RequestInit = {
      method: "POST"
    };
    if (signal) {
      opts.signal = signal;
    }
    await this.request<{ status: string }>("/dara/shutdown", opts).catch(() => {});
  }

  async streamChat(
    req: ChatRequest,
    signal?: AbortSignal
  ): Promise<AsyncGenerator<ChatChunk, void, unknown>> {
    const url = `${this.baseUrl}/v1/chat/completions`;
    const headers = new Headers({
      "Content-Type": "application/json",
      Accept: "text/event-stream"
    });
    if (this.apiKey) {
      headers.set("Authorization", `Bearer ${this.apiKey}`);
    }

    const fetchOpts: RequestInit = {
      method: "POST",
      headers,
      body: JSON.stringify({ ...req, stream: true })
    };
    if (signal) {
      fetchOpts.signal = signal;
    }

    const response = await fetch(url, fetchOpts);

    if (!response.ok) {
      let errorBody: unknown;
      try {
        errorBody = await response.json();
      } catch {
        errorBody = await response.text();
      }
      throw ApiError.fromResponseBody(response.status, errorBody, "Chat completion failed");
    }

    if (!response.body) {
      throw new ApiError(0, "Empty response body from chat completion stream", "empty_response");
    }

    return parseSseStream(response.body.getReader());
  }

  /** One non-streaming chat completion (used for context-window summaries). */
  async chat(req: ChatRequest, signal?: AbortSignal): Promise<ChatResponse> {
    const opts: RequestInit & { timeoutMs: number } = {
      method: "POST",
      body: JSON.stringify({ ...req, stream: false }),
      timeoutMs: 600000
    };
    if (signal) opts.signal = signal;
    return this.request<ChatResponse>("/v1/chat/completions", opts);
  }

  async getModelOptions(
    modelId: string,
    signal?: AbortSignal,
    query = ""
  ): Promise<OptionsResponse> {
    const opts: RequestInit = {
      method: "GET"
    };
    if (signal) {
      opts.signal = signal;
    }
    return this.request<OptionsResponse>(`/dara/models/${modelId}/options${query}`, opts);
  }

  async startBenchmark(
    modelId: string,
    body: { formats?: string[]; vision?: boolean; force?: boolean } = {},
    signal?: AbortSignal
  ): Promise<BenchmarkJob> {
    const opts: RequestInit = {
      method: "POST",
      body: JSON.stringify(body)
    };
    if (signal) {
      opts.signal = signal;
    }
    return this.request<BenchmarkJob>(`/dara/models/${modelId}/benchmark`, opts);
  }

  async getBenchmarkJob(jobId: string, signal?: AbortSignal): Promise<BenchmarkJob> {
    const opts: RequestInit = {
      method: "GET"
    };
    if (signal) {
      opts.signal = signal;
    }
    return this.request<BenchmarkJob>(`/dara/benchmarks/${jobId}`, opts);
  }

  async cancelBenchmark(jobId: string, signal?: AbortSignal): Promise<void> {
    const opts: RequestInit = {
      method: "DELETE"
    };
    if (signal) {
      opts.signal = signal;
    }
    await this.request<undefined>(`/dara/benchmarks/${jobId}`, opts);
  }
}

// Global default singleton instance
export const apiClient = new DaraApiClient();
