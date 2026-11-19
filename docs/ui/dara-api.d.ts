// TypeScript types of the Dara server API (docs/ui/04-backend-api.md). Copy to app/src/lib/api/types.ts.
// Derived from real responses in docs/ui/samples. Fields marked optional may be absent; unknown fields must be tolerated.

export type KvFormat = "q8" | "k8v4" | "q4" | "tq4" | "k16v8";
export type KvChoice = "auto" | KvFormat;
export type QualityLabel = "high" | "medium" | "experimental" | "lower";

export interface ApiErrorBody {
  error: { message: string; type: string; code?: string; download?: { model: string; [k: string]: unknown } };
}

export interface Health {
  status: "ok" | "loading" | "error";
  model?: string;
  busy: boolean;
  queued: number;
  loading?: unknown;
}

export interface VersionInfo { version: string; api: number; built: string; calibration_file: string }

// ---- models ----
export interface ModelFile { kind: "model" | "mmproj" | "flash"; name: string; installed: boolean; size_mb: number; url: string }
export interface Feature { available: boolean; installed: boolean; active: boolean; note?: string }
export interface LoadedConfig {
  kv: KvFormat; flash: boolean; context: number; layers: number; layers_on_gpu: number; head: "gpu" | "cpu"; vision: boolean;
  source: "recommended" | "user" | string; reason: string;
}
export interface ModelInfo {
  id: string;
  object: "model";
  owned_by: "dara";
  created: number;
  dara: {
    name: string; base_model: string; description: string; family: "gemma4" | "qwen35" | string; quant: string;
    status: "installed" | "partial" | "not_installed" | "downloading";
    loaded: boolean; fits_gpu: "full" | "partial" | "no" | string;
    min_vram_mb: number; full_vram_mb: number; context: number;
    download_size_mb: number; download: DownloadJob | null;
    files: ModelFile[]; flash: Feature; vision: Feature;
    config?: LoadedConfig; // present when loaded
  };
}
export interface ModelList { object: "list"; data: ModelInfo[]; models_dir: string }

export interface DownloadJob {
  id: string; model: string; status: string; file: string; files_done: number; files_total: number; completed_bytes: number; total_bytes: number;
  percent: number; mb_per_second: number; eta_seconds: number; seconds: number; error?: string;
}
export interface PullRequest { model: string; files?: Array<"model" | "mmproj" | "flash">; stream?: boolean }
export interface LoadRequest { model: string; kv?: KvChoice; flash?: "auto" | boolean; vision?: boolean; ctx?: number; async?: boolean }

export interface LoadStatus {
  state: "idle" | "queued" | "loading" | "ready" | "failed";
  stage?: string; percent?: number; done?: number; total?: number; model?: string; seconds?: number; error?: string;
  config?: Partial<LoadedConfig>;
}

// ---- options ----
export interface QualityInfo {
  label: QualityLabel; text: string; checked_on: string; min_top1_of_128: number; worst_clear_errors_of_128: number; worst_kl: number;
}
export interface FormatInfo {
  id: KvFormat; name: string; description: string; caveat: string; bits_per_value: number; recommendable: boolean; quality: QualityInfo;
}
export type Speed =
  | { source: "not_measured" }
  | { source: "measured"; decode_tps_start: number; decode_tps_full_estimate?: number; prefill_tps?: number; measured_at?: string; power?: "ac" | "battery" | "unknown" };
export interface Placement {
  context: number; requested_context?: number; layers: number; layers_gpu: number; head: "gpu" | "cpu";
  fully_on_gpu: boolean; all_layers_on_gpu: boolean; flash_active: boolean; kv_cache_mb: number; ok: boolean; speed: Speed;
}
export interface OptionEntry { kv: KvFormat; flash: boolean; fast: Placement; points: Placement[] }
export interface OptionsResponse {
  model: string; name: string; family: string; native_context: number;
  machine: { gpu: string; vram_total_mb: number; vram_free_assumed_mb: number; driver: number; dara: string };
  vision: { included: boolean; requested: boolean; reserved_mb: number; note: string };
  flash: unknown;
  formats: FormatInfo[];
  options: OptionEntry[];
  recommended: { kv: KvFormat; flash: boolean; context: number; reason: string; rule: string };
  calibration: {
    status: "none" | "partial" | "measured" | "stale"; entries: number; entries_from_other_machines: number; entries_measured_on_battery: number;
    fast_options: number; fast_options_measured: number; note: string; run: { method: string; url: string };
  };
  notes: string[];
}
export interface BenchmarkJob { id: string; status: "running" | "done" | "failed" | "cancelled"; log: string[]; error?: string }

// ---- chat ----
export type ContentPart = { type: "text"; text: string } | { type: "image_url"; image_url: { url: string } };
export interface ChatMessage { role: "system" | "user" | "assistant"; content: string | ContentPart[] }
export interface ChatRequest {
  model?: string; messages: ChatMessage[]; stream?: boolean; max_tokens?: number; temperature?: number; top_k?: number; top_p?: number;
  min_p?: number; repeat_penalty?: number; seed?: number; stop?: string | string[]; flash?: boolean; cache_write?: boolean;
}
export interface Usage { prompt_tokens: number; completion_tokens: number; total_tokens: number; prompt_tokens_details: { cached_tokens: number; image_tokens: number } }
export interface Timings {
  queue_ms: number; compact_ms: number; image_ms: number; prefill_tokens: number; prefill_ms: number; prefill_tokens_per_second: number;
  ttft_ms: number; decode_tokens: number; decode_ms: number; decode_tokens_per_second: number; total_ms: number;
  image_memory: { scratch_mb: number; from_unused_context_mb: number; from_free_vram_mb: number };
}
export interface ContextInfo {
  capacity: number; used: number; compacted: boolean; summarized_messages: number; dropped_messages: number; truncated_last_message: boolean; summary_cached: boolean;
}
export interface SpeculativeInfo {
  mode: string; verify_steps: number; drafted: number; accepted: number; acceptance: number; model_passes: number; tokens_per_pass: number;
  note: string | null; flash_stood_down_at_pass: number | null;
}
export interface ChatStats { usage: Usage; timings: Timings; context: ContextInfo; speculative?: SpeculativeInfo }
export interface ChatResponse extends ChatStats {
  id: string; object: "chat.completion"; created: number; model: string;
  choices: Array<{ index: number; finish_reason: "stop" | "length" | string; message: { role: "assistant"; content: string } }>;
}
export interface ChatChunk extends Partial<ChatStats> {
  id: string; object: "chat.completion.chunk"; created: number; model: string;
  choices: Array<{ index: number; finish_reason: string | null; delta: { role?: "assistant"; content?: string } }>;
}

// ---- stats ----
export interface Stats {
  uptime_seconds: number; pid: number; queued: number;
  context: { capacity: number; held_tokens: number };
  gpu?: {
    name: string; nvml: boolean; compute_capability: string; vram_total_mb: number; vram_used_mb: number; vram_free_mb: number; vram_used_percent: number; vram_weights_mb: number;
    utilization_percent?: number; memory_utilization_percent?: number; temperature_c?: number; power_w?: number; power_limit_w?: number; clock_sm_mhz?: number; clock_mem_mhz?: number; pstate?: number;
  };
  engine: {
    requests: number; errors: number; busy: boolean; live_tokens: number; live_tokens_per_second: number; avg_decode_tokens_per_second: number; avg_prefill_tokens_per_second: number;
    compactions: number; prompt_tokens_total: number; completion_tokens_total: number; cached_tokens_total: number;
  };
  model?: Record<string, unknown> & { id: string; kv_format: KvFormat; layers: number; layers_on_gpu: number; output_head: "gpu" | "cpu"; load_seconds: number; vision: boolean };
  disk: { models_dir: string; volume_free_gb: number; volume_total_gb: number; installed_models_mb: number; model_file_mb: number; mmproj_file_mb: number; flash_file_mb: number };
  downloads: DownloadJob[];
  last_request: (ChatStats & Record<string, unknown>) | null;
  ram?: Record<string, number>; process?: Record<string, number>;
}
