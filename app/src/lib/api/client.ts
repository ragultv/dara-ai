import { ApiError } from "./errors.ts";
import type { Health, ModelList } from "./types.ts";

export interface ClientConfig {
  baseUrl: string;
  apiKey?: string | undefined;
  defaultTimeoutMs?: number | undefined;
}

export class DaraApiClient {
  private baseUrl: string;
  private apiKey?: string | undefined;
  private defaultTimeoutMs: number;

  constructor(config: ClientConfig = { baseUrl: "http://127.0.0.1:11435" }) {
    this.baseUrl = config.baseUrl.replace(/\/+$/, "");
    this.apiKey = config.apiKey;
    this.defaultTimeoutMs = config.defaultTimeoutMs ?? 30000;
  }

  setConnection(baseUrl: string, apiKey?: string | undefined): void {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.apiKey = apiKey;
  }

  getBaseUrl(): string {
    return this.baseUrl;
  }

  async request<T>(
    endpoint: string,
    options: RequestInit & { timeoutMs?: number | undefined } = {}
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

      if (!response.ok) {
        let errorBody: unknown;
        try {
          errorBody = await response.json();
        } catch {
          errorBody = await response.text();
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
}

// Global default singleton instance
export const apiClient = new DaraApiClient();
