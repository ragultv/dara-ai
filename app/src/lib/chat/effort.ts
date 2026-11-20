/**
 * Thinking effort. The names are the OpenAI / Hermes / Claude Code ones; the server turns each into a cap on thinking tokens
 * (src/serve/server.cpp `effort_budget`). When the cap runs out the engine closes the thought and the model answers.
 * Keep the numbers here in step with the server: they are only shown to the user.
 */
export type ReasoningEffort = "none" | "low" | "medium" | "high" | "xhigh" | "max";

export interface EffortLevel {
  id: ReasoningEffort;
  label: string;
  /** Thinking-token cap the server applies; 0 = no thinking. */
  budget: number;
  description: string;
}

export const EFFORT_LEVELS: EffortLevel[] = [
  { id: "none", label: "Off", budget: 0, description: "No thinking. The fastest replies." },
  { id: "low", label: "Low", budget: 512, description: "A short think, up to about 500 tokens." },
  {
    id: "medium",
    label: "Medium",
    budget: 2048,
    description: "Works through the problem, up to about 2,000 tokens."
  },
  {
    id: "high",
    label: "High",
    budget: 4096,
    description: "Careful step-by-step reasoning, up to about 4,000 tokens."
  },
  {
    id: "xhigh",
    label: "Extra",
    budget: 8192,
    description: "Deep reasoning for hard problems, up to about 8,000 tokens."
  },
  {
    id: "max",
    label: "Max",
    budget: 32768,
    description: "As long as it needs, up to about 32,000 tokens. Can be slow."
  }
];

export function effortLevel(id: ReasoningEffort): EffortLevel {
  return EFFORT_LEVELS.find((l) => l.id === id) ?? EFFORT_LEVELS[0]!;
}

export function isEffort(v: unknown): v is ReasoningEffort {
  return typeof v === "string" && EFFORT_LEVELS.some((l) => l.id === v);
}

/** "12s", "1m 05s" for the Reasoning header. */
export function formatThinkingTime(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
}
