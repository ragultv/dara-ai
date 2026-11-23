// Facts the site states. Everything here comes from the repository, not from marketing: model files and sizes from src/serve/catalog.cpp, minimum VRAM from the same
// catalog, how the app is installed from docs/RELEASE.md, the API from docs/SERVER.md. Measured numbers are NOT here: they are generated into benchmarks.json
// by site/scripts/make_data.py from the recorded benchmark runs.

export const REPO = "https://github.com/ragultv/dara-ai";
export const RELEASES = `${REPO}/releases`;
export const DOCS = `${REPO}/tree/dev/docs`;

export interface ModelInfo {
  id: string;
  name: string;
  base: string;
  blurb: string;
  modelGb: number;       // the GGUF
  visionGb: number;      // the image encoder, downloaded with it
  drafter: string | null;
  minVramGb: number;
  fullVramGb: number;
  note: string;
}

export const MODELS: ModelInfo[] = [
  {
    id: "dara-nano",
    name: "Dara Nano",
    base: "Gemma 4 E2B (instruct)",
    blurb: "The smallest and fastest tier: chat, quick answers, image understanding.",
    modelGb: 3.11,
    visionGb: 0.99,
    drafter: "Flash drafter v3 (37 MB)",
    minVramGb: 3,
    fullVramGb: 4,
    note: "On our 4 GB test laptop the engine fits all 35 layers and the model's full 131,072-token context on the GPU."
  },
  {
    id: "dara-core-4b",
    name: "Dara Core 4B",
    base: "Qwen3.5 4B",
    blurb: "A balanced tier: reasoning, coding, documents, image understanding.",
    modelGb: 2.74,
    visionGb: 0.67,
    drafter: "Flash drafter v2 (86 MB)",
    minVramGb: 3,
    fullVramGb: 4,
    note: "Fits a 4 GB GPU. The app shows how much context fits your GPU before you load it."
  },
  {
    id: "dara-pro-9b",
    name: "Dara Pro 9B",
    base: "Qwen3.5 9B",
    blurb: "The most capable tier: harder reasoning and coding.",
    modelGb: 5.68,
    visionGb: 0.92,
    drafter: null,
    minVramGb: 4,
    fullVramGb: 8,
    note: "Runs partly on the CPU on GPUs under 8 GB, so it is noticeably slower there. No Flash drafter yet."
  }
];
