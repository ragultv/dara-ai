import React, { useEffect, useState } from "react";
import { useEngineStore, type EngineSettingsValues } from "../../stores/useEngineStore.ts";
import { useSettingsStore } from "../../stores/useSettingsStore.ts";
import { Button } from "../../ui/Button.tsx";
import { Dialog } from "../../ui/Dialog.tsx";
import { toast } from "../../ui/Toast.tsx";
import { ConnectionInfo } from "./ConnectionInfo.tsx";
import { EngineSettingsForm } from "./EngineSettingsForm.tsx";
import { SettingsSection } from "./SettingsSection.tsx";
import type { ServerFacts } from "./useServerFacts.ts";

interface EngineSectionProps {
  facts: ServerFacts;
}

export const EngineSection: React.FC<EngineSectionProps> = ({ facts }) => {
  const isAvailable = useEngineStore((s) => s.isAvailable);
  const info = useEngineStore((s) => s.info);
  const isRestarting = useEngineStore((s) => s.isRestarting);
  const error = useEngineStore((s) => s.error);
  const load = useEngineStore((s) => s.load);
  const save = useEngineStore((s) => s.save);
  const restart = useEngineStore((s) => s.restart);
  const updateServer = useSettingsStore((s) => s.updateServer);
  const updateInference = useSettingsStore((s) => s.updateInference);

  const [confirm, setConfirm] = useState<"restart" | "regenerate" | null>(null);

  useEffect(() => {
    void load();
  }, [load]);

  const handleSave = async (values: EngineSettingsValues): Promise<boolean> => {
    const ok = await save(values);
    if (ok) {
      // Mirror into the database so the rest of the app sees the same values.
      updateServer({
        port: values.port,
        modelsDir: values.models_dir,
        allowNetworkAccess: values.allow_lan
      });
      updateInference({ vramReserveMb: values.vram_reserve_mb });
      toast.show("Engine settings saved. Restart the engine to apply them.", "success");
    }
    return ok;
  };

  const runRestart = async (regenerate: boolean): Promise<void> => {
    setConfirm(null);
    const ok = await restart(regenerate);
    if (ok) {
      toast.show(
        regenerate ? "Engine restarted with a new API key." : "Engine restarted.",
        "success"
      );
      facts.refresh();
    } else {
      toast.show("The engine could not be restarted. See the message in Settings.", "error");
    }
  };

  return (
    <SettingsSection
      id="settings-engine"
      icon="cpu"
      title="Engine"
      description="The program that runs your models. Changes here apply after the engine restarts."
    >
      <ConnectionInfo
        facts={facts}
        onRegenerate={isAvailable ? () => setConfirm("regenerate") : undefined}
      />

      <div className="space-y-3 border-t border-border pt-5">
        <h3 className="text-sm font-semibold text-text">Engine settings</h3>
        {!isAvailable && (
          <p
            role="note"
            className="rounded-xl border border-border bg-surface-sunken p-3 text-xs text-text-muted"
          >
            These settings are applied by the Dara desktop app when it starts the engine. This
            window is running outside the app (browser or mock mode), so they are disabled.
          </p>
        )}
        {isAvailable && info && (
          <EngineSettingsForm info={info} isBusy={isRestarting} onSave={handleSave} />
        )}
        {isAvailable && !info && !error && (
          <p className="text-xs text-text-muted">Reading engine settings...</p>
        )}

        {error && (
          <p role="alert" className="text-xs font-medium text-danger">
            {error}
          </p>
        )}

        {isAvailable && info?.restart_needed && (
          <div
            role="status"
            className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-primary/30 bg-primary-subtle p-3"
          >
            <p className="text-xs font-medium text-on-primary-subtle">
              Restart the engine to apply your saved engine settings.
            </p>
            <Button
              variant="primary"
              size="sm"
              disabled={isRestarting}
              onClick={() => setConfirm("restart")}
            >
              {isRestarting ? "Restarting..." : "Restart engine"}
            </Button>
          </div>
        )}
        {isAvailable && info && !info.restart_needed && (
          <div className="flex justify-end">
            <Button
              variant="ghost"
              size="xs"
              disabled={isRestarting}
              onClick={() => setConfirm("restart")}
            >
              Restart engine
            </Button>
          </div>
        )}
      </div>

      <Dialog
        isOpen={confirm !== null}
        title={confirm === "regenerate" ? "Generate a new API key?" : "Restart the engine?"}
        description={
          confirm === "regenerate"
            ? "The engine restarts with a new key. Programs that use the old key stop working until you give them the new one."
            : "The engine stops and starts again with your saved settings."
        }
        onclose={() => setConfirm(null)}
        actions={
          <>
            <Button variant="ghost" size="sm" onClick={() => setConfirm(null)}>
              Cancel
            </Button>
            <Button
              variant="primary"
              size="sm"
              onClick={() => void runRestart(confirm === "regenerate")}
            >
              Restart engine
            </Button>
          </>
        }
      >
        <p>
          The loaded model is unloaded, and the engine starts again with its own default model; load
          another from the Models page if you want a different one. Your chats are saved and keep
          working once the engine is back. A reply that is being written right now will stop.
        </p>
      </Dialog>
    </SettingsSection>
  );
};
