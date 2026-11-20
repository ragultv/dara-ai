import { create } from "zustand";
import { isEffort, type ReasoningEffort } from "../chat/effort.ts";

// The chosen thinking effort is a per-device choice (like the composer's other toggles), kept in localStorage. Off is the default:
// thinking costs time, so the user opts in.
const KEY = "dara_thinking_effort";

function read(): ReasoningEffort {
  try {
    const v = localStorage.getItem(KEY);
    if (isEffort(v)) return v;
  } catch {
    // ignore: optional
  }
  return "none";
}

interface EffortState {
  effort: ReasoningEffort;
  setEffort: (e: ReasoningEffort) => void;
}

export const useEffortStore = create<EffortState>((set) => ({
  effort: read(),
  setEffort: (effort) => {
    try {
      localStorage.setItem(KEY, effort);
    } catch {
      // ignore
    }
    set({ effort });
  }
}));
