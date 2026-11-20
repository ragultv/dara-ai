import React, { useState } from "react";
import { Icon } from "../../ui/Icon.tsx";
import { Dialog } from "../../ui/Dialog.tsx";
import { Button } from "../../ui/Button.tsx";
import { toast } from "../../ui/Toast.tsx";
import { useChatStore } from "../../stores/useChatStore.ts";
import { useModelsStore } from "../../stores/useModelsStore.ts";
import { addonSizeMb, addonState, enableAddon, useAddonProgress } from "../../chat/addons.ts";

/**
 * Flash (speculative decoding) as the engine really has it:
 *  - active: the drafter is loaded; this switch turns it on/off per request (no reload, the context does not change).
 *  - installed but not loaded, or not installed: turning it on reloads the model with it. Flash needs memory, so the context the engine
 *    can offer may shrink; the real size is read back from the engine and shown.
 */
export const FlashToggle: React.FC<{ className?: string }> = ({ className = "" }) => {
  const activeConv = useChatStore(
    (s) => s.conversations.find((c) => c.id === s.activeConversationId) ?? null
  );
  const setConversationSettings = useChatStore((s) => s.setConversationSettings);
  const isGenerating = useChatStore((s) => s.isGenerating);
  const model = useModelsStore((s) => s.models.find((m) => m.dara.loaded) ?? null);
  const busy = useAddonProgress((s) => s.busy);
  const [confirmOpen, setConfirmOpen] = useState(false);

  const state = addonState(model, "flash");
  const sizeMb = addonSizeMb(model, "flash");
  const requested = activeConv?.flash !== false;

  if (!model) return null;

  const pillBase =
    "h-6 px-2.5 rounded-full border transition-all flex items-center gap-1.5 text-xs font-medium select-none";

  if (state === "unavailable") {
    return (
      <span
        className={`${pillBase} bg-surface-sunken border-border text-text-subtle ${className}`}
        title={`${model.dara.name} has no Flash drafter yet`}
      >
        <Icon name="zap" size={11} />
        <span className="text-[11px] font-semibold">Flash</span>
        <span className="text-[9px] uppercase tracking-wider font-bold opacity-80">N/A</span>
      </span>
    );
  }

  const runEnable = async () => {
    setConfirmOpen(false);
    try {
      const cfg = await enableAddon(model.id, "flash");
      if (cfg?.flash) {
        toast.show(
          `Flash is on. The context is now ${cfg.context.toLocaleString()} tokens.`,
          "success"
        );
        if (activeConv) setConversationSettings(activeConv.id, { flash: true });
      }
    } catch (err) {
      toast.show(err instanceof Error ? err.message : "Could not turn Flash on", "error");
    }
  };

  const isOn = state === "active" && requested;
  const label = state === "active" ? (isOn ? "ON" : "OFF") : state === "install" ? "GET" : "OFF";
  const title =
    state === "active"
      ? isOn
        ? "Flash is on: faster generation. Click to turn it off for this chat."
        : "Flash is off for this chat. Click to turn it on."
      : state === "install"
        ? "Flash drafter is not installed. Click to download it and reload the model."
        : "Flash is installed but the model was loaded without it. Click to reload with Flash.";

  return (
    <>
      <button
        type="button"
        disabled={busy || isGenerating}
        onClick={() => {
          if (state === "active") {
            if (activeConv) setConversationSettings(activeConv.id, { flash: !requested });
          } else {
            setConfirmOpen(true);
          }
        }}
        className={`${pillBase} cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed ${
          isOn
            ? "bg-text/15 border-border-strong text-text shadow-xs hover:bg-text/25"
            : "bg-surface-sunken border-border text-text-subtle hover:text-text hover:bg-surface-raised"
        } ${className}`}
        title={title}
        aria-label="Flash speculative decoding"
        aria-pressed={isOn}
      >
        <Icon name="zap" size={11} className={isOn ? "text-text" : "text-text-subtle"} />
        <span className="text-[11px] font-semibold">Flash</span>
        <span className="text-[9px] uppercase tracking-wider font-bold opacity-80">{label}</span>
      </button>

      <Dialog
        isOpen={confirmOpen}
        title="Turn on Flash?"
        description="Flash drafts several tokens at once and checks them with the full model: same answers, faster generation."
        onclose={() => setConfirmOpen(false)}
        actions={
          <>
            <Button variant="ghost" size="sm" onClick={() => setConfirmOpen(false)}>
              Not now
            </Button>
            <Button variant="primary" size="sm" onClick={() => void runEnable()}>
              {state === "install" ? "Download and turn on" : "Reload with Flash"}
            </Button>
          </>
        }
      >
        <ul className="text-xs text-text-muted space-y-1.5 list-disc pl-4 py-1">
          {state === "install" && (
            <li>Downloads the Flash drafter{sizeMb ? ` (${Math.round(sizeMb)} MB)` : ""}.</li>
          )}
          <li>Reloads the model, which takes a few seconds. Your conversations are kept.</li>
          <li>
            Flash uses GPU memory, so the largest context the engine can offer may be smaller. The
            real size is shown afterwards.
          </li>
          <li>It is only kept on when the engine can still fit a long context in memory.</li>
        </ul>
      </Dialog>
    </>
  );
};
