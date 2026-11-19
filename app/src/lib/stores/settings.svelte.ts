export interface GeneralSettings {
  startAtLogin: boolean;
  minimizeToTray: boolean;
}

export interface ServerSettings {
  port: number;
  modelsDir: string;
  allowNetworkAccess: boolean;
  apiKey: string;
}

export interface InferenceDefaults {
  defaultKv: "auto" | "q8" | "k8v4" | "q4";
  defaultFlash: "auto" | "on" | "off";
  vramReserveMb: number;
  visionEnabled: boolean;
  defaultTemperature: number;
  defaultMaxTokens: number;
  systemPrompt: string;
}

export interface AppSettings {
  general: GeneralSettings;
  server: ServerSettings;
  inference: InferenceDefaults;
}

const STORAGE_KEY = "dara_settings";

const defaultSettings: AppSettings = {
  general: {
    startAtLogin: false,
    minimizeToTray: true
  },
  server: {
    port: 11435,
    modelsDir: "D:\\darap-ai\\models",
    allowNetworkAccess: false,
    apiKey: "dara-key-" + Math.random().toString(36).substring(2, 10)
  },
  inference: {
    defaultKv: "auto",
    defaultFlash: "auto",
    vramReserveMb: 64,
    visionEnabled: true,
    defaultTemperature: 0.7,
    defaultMaxTokens: 4096,
    systemPrompt:
      "You are a helpful, respectful AI assistant running locally on the user's hardware."
  }
};

class SettingsStore {
  general = $state<GeneralSettings>({ ...defaultSettings.general });
  server = $state<ServerSettings>({ ...defaultSettings.server });
  inference = $state<InferenceDefaults>({ ...defaultSettings.inference });

  constructor() {
    this.load();
  }

  load(): void {
    if (typeof window === "undefined" || !window.localStorage) return;
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw) as Partial<AppSettings>;
        if (parsed.general) this.general = { ...defaultSettings.general, ...parsed.general };
        if (parsed.server) this.server = { ...defaultSettings.server, ...parsed.server };
        if (parsed.inference)
          this.inference = { ...defaultSettings.inference, ...parsed.inference };
      }
    } catch {
      // fallback to defaults on corrupt storage
    }
  }

  save(): void {
    if (typeof window === "undefined" || !window.localStorage) return;
    try {
      const payload: AppSettings = {
        general: this.general,
        server: this.server,
        inference: this.inference
      };
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
    } catch {
      // storage unavailable
    }
  }

  updateGeneral(patch: Partial<GeneralSettings>): void {
    this.general = { ...this.general, ...patch };
    this.save();
  }

  updateServer(patch: Partial<ServerSettings>): void {
    this.server = { ...this.server, ...patch };
    this.save();
  }

  updateInference(patch: Partial<InferenceDefaults>): void {
    this.inference = { ...this.inference, ...patch };
    this.save();
  }

  regenerateApiKey(): string {
    const newKey = "dara-key-" + Math.random().toString(36).substring(2, 10);
    this.server.apiKey = newKey;
    this.save();
    return newKey;
  }

  reset(): void {
    this.general = { ...defaultSettings.general };
    this.server = { ...defaultSettings.server };
    this.inference = { ...defaultSettings.inference };
    this.save();
  }
}

export const settingsStore = new SettingsStore();
