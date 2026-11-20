import React from "react";
import { useSettingsStore, type InferenceSettings } from "../../stores/useSettingsStore.ts";
import type { KvChoice } from "../../api/types.ts";
import { Button } from "../../ui/Button.tsx";
import { SegmentedControl } from "../../ui/SegmentedControl.tsx";
import { Slider } from "../../ui/Slider.tsx";
import { Textarea } from "../../ui/Textarea.tsx";
import { Toggle } from "../../ui/Toggle.tsx";
import { toast } from "../../ui/Toast.tsx";
import { NumberField } from "./NumberField.tsx";
import { SettingsSection } from "./SettingsSection.tsx";

const KV_OPTIONS = [
  { value: "auto", label: "Auto" },
  { value: "q8", label: "Q8" },
  { value: "k8v4", label: "K8V4" },
  { value: "q4", label: "Q4" }
];

const FLASH_OPTIONS = [
  { value: "auto", label: "Auto" },
  { value: "on", label: "On" },
  { value: "off", label: "Off" }
];

export const DefaultsSection: React.FC = () => {
  const inference = useSettingsStore((s) => s.inference);
  const updateInference = useSettingsStore((s) => s.updateInference);
  const reset = useSettingsStore((s) => s.reset);

  return (
    <SettingsSection
      id="settings-defaults"
      icon="sliders-horizontal"
      title="Defaults"
      description="Starting values. Each chat and each model load can still change them."
    >
      <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
        <div className="space-y-1.5">
          <SegmentedControl
            label="K/V format"
            options={KV_OPTIONS}
            value={inference.defaultKv}
            onchange={(v) => updateInference({ defaultKv: v as KvChoice })}
          />
          <p className="text-[11px] text-text-muted">
            How the conversation memory is stored on the GPU. Auto lets Dara pick; the Options page
            of a model explains each choice. Used when you load a model from the app.
          </p>
        </div>
        <div className="space-y-1.5">
          <SegmentedControl
            label="Flash"
            options={FLASH_OPTIONS}
            value={inference.defaultFlash}
            onchange={(v) =>
              updateInference({ defaultFlash: v as InferenceSettings["defaultFlash"] })
            }
          />
          <p className="text-[11px] text-text-muted">
            Faster replies using a small helper model; it takes some video memory, so the context
            gets shorter. Used when you load a model from the app.
          </p>
        </div>
      </div>

      <div className="border-t border-border pt-5">
        <Toggle
          label="Image support"
          description="Load the part of the model that reads images when a model has it. Without it, image attachments are disabled."
          checked={inference.visionEnabled}
          onChange={(v) => updateInference({ visionEnabled: v })}
        />
      </div>

      <div className="space-y-1.5 border-t border-border pt-5">
        <Slider
          label="Temperature"
          min={0}
          max={2}
          step={0.1}
          value={inference.defaultTemperature}
          onChange={(v) => updateInference({ defaultTemperature: Math.round(v * 10) / 10 })}
        />
        <p className="text-[11px] text-text-muted">
          Lower is steadier and more repeatable (0.2); higher is more varied (1.0). Used by new
          chats.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
        <NumberField
          label="Max tokens per reply"
          min={1}
          max={1_000_000}
          value={inference.defaultMaxTokens}
          hint="A reply stops at this length (a token is about three quarters of a word), or earlier at the end of the context."
          onCommit={(n) => updateInference({ defaultMaxTokens: n })}
        />
      </div>

      <div className="space-y-1.5 border-t border-border pt-5">
        <Textarea
          label="System prompt"
          rows={4}
          value={inference.systemPrompt}
          placeholder="Optional instructions that every new chat starts with"
          hint="Sent at the start of every new chat. Leave empty for none."
          onChange={(e) => updateInference({ systemPrompt: e.target.value })}
        />
      </div>

      <div className="flex justify-end">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => {
            reset();
            toast.show("Defaults restored.", "info");
          }}
        >
          Restore defaults
        </Button>
      </div>
    </SettingsSection>
  );
};
