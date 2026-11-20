import React, { useEffect, useState } from "react";
import type { EngineSettingsInfo, EngineSettingsValues } from "../../stores/useEngineStore.ts";
import { Button } from "../../ui/Button.tsx";
import { Input } from "../../ui/Input.tsx";
import { Toggle } from "../../ui/Toggle.tsx";

interface EngineSettingsFormProps {
  info: EngineSettingsInfo;
  isBusy: boolean;
  onSave: (values: EngineSettingsValues) => Promise<boolean>;
}

interface Draft {
  port: string;
  modelsDir: string;
  allowLan: boolean;
  vramReserve: string;
}

function toDraft(v: EngineSettingsValues): Draft {
  return {
    port: String(v.port),
    modelsDir: v.models_dir,
    allowLan: v.allow_lan,
    vramReserve: String(v.vram_reserve_mb)
  };
}

function wholeNumber(text: string): number | null {
  const n = Number(text);
  return text.trim() !== "" && Number.isInteger(n) ? n : null;
}

/** Pure checks that mirror the Rust validation; Rust still has the last word (port in use, folder access). */
export function validateDraft(d: Draft): { port?: string; vram?: string; folder?: string } {
  const errors: { port?: string; vram?: string; folder?: string } = {};
  const port = wholeNumber(d.port);
  if (port === null || port < 1024 || port > 65535)
    errors.port = "Enter a port from 1024 to 65535.";
  const vram = wholeNumber(d.vramReserve);
  if (vram === null || vram < 0 || vram > 8192)
    errors.vram = "Enter a number of MB from 0 to 8192.";
  const dir = d.modelsDir.trim();
  if (dir !== "" && !/^([a-zA-Z]:[\\/]|\\\\|\/)/.test(dir)) {
    errors.folder = "Use a full path such as D:\\models, or leave it empty for the default folder.";
  }
  return errors;
}

export const EngineSettingsForm: React.FC<EngineSettingsFormProps> = ({ info, isBusy, onSave }) => {
  const [draft, setDraft] = useState<Draft>(() => toDraft(info.saved));

  useEffect(() => {
    setDraft(toDraft(info.saved));
  }, [info.saved]);

  const errors = validateDraft(draft);
  const hasErrors = Object.keys(errors).length > 0;
  const saved = toDraft(info.saved);
  const isDirty =
    draft.port !== saved.port ||
    draft.modelsDir.trim() !== saved.modelsDir ||
    draft.allowLan !== saved.allowLan ||
    draft.vramReserve !== saved.vramReserve;

  const submit = async (): Promise<void> => {
    if (hasErrors) return;
    await onSave({
      port: Number(draft.port),
      models_dir: draft.modelsDir.trim(),
      allow_lan: draft.allowLan,
      vram_reserve_mb: Number(draft.vramReserve)
    });
  };

  return (
    <form
      className="space-y-4"
      onSubmit={(e) => {
        e.preventDefault();
        void submit();
      }}
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Input
          label="Port"
          inputMode="numeric"
          value={draft.port}
          onChange={(e) => setDraft({ ...draft, port: e.target.value })}
          {...(errors.port
            ? { error: errors.port }
            : { hint: "Where the engine listens. Default 11435." })}
        />
        <Input
          label="VRAM reserve (MB)"
          inputMode="numeric"
          value={draft.vramReserve}
          onChange={(e) => setDraft({ ...draft, vramReserve: e.target.value })}
          {...(errors.vram
            ? { error: errors.vram }
            : {
                hint: "Video memory Dara leaves free when it works out how much context fits. Raise it if other programs run short of VRAM."
              })}
        />
      </div>

      <Input
        label="Models folder"
        value={draft.modelsDir}
        placeholder={info.default_models_dir}
        onChange={(e) => setDraft({ ...draft, modelsDir: e.target.value })}
        {...(errors.folder
          ? { error: errors.folder }
          : {
              hint: "Where model files are stored. Empty uses the default shown. Existing models are not moved."
            })}
      />

      <Toggle
        label="Allow other devices on my network"
        description="Lets phones and other computers on your Wi-Fi reach the engine. They must send the API key."
        checked={draft.allowLan}
        onChange={(v) => setDraft({ ...draft, allowLan: v })}
      />
      {draft.allowLan && (
        <div
          role="alert"
          className="rounded-xl border border-warning/40 bg-warning-subtle p-3 text-xs text-text"
        >
          <p className="font-semibold">
            Anyone on your network who has the key can use your models.
          </p>
          <p className="mt-1 text-text-muted">
            The engine will listen on every network adapter instead of only this computer. Share the
            key only with devices you trust, and keep this off on public Wi-Fi.
          </p>
        </div>
      )}

      <div className="flex items-center gap-2">
        <Button
          type="submit"
          variant="primary"
          size="sm"
          disabled={!isDirty || hasErrors || isBusy}
        >
          Save engine settings
        </Button>
        {isDirty && (
          <Button
            variant="ghost"
            size="sm"
            type="button"
            onClick={() => setDraft(toDraft(info.saved))}
          >
            Discard changes
          </Button>
        )}
      </div>
    </form>
  );
};
