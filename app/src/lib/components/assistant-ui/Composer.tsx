import React, { useState, useRef, useEffect, useCallback } from "react";
import { Icon } from "../../ui/Icon.tsx";
import { Popover } from "../../ui/Popover.tsx";
import { Slider } from "../../ui/Slider.tsx";
import { Textarea } from "../../ui/Textarea.tsx";
import { Button } from "../../ui/Button.tsx";
import { Dialog } from "../../ui/Dialog.tsx";
import { ProgressBar } from "../../ui/ProgressBar.tsx";
import { NumberTicker } from "../../ui/NumberTicker.tsx";
import { toast } from "../../ui/Toast.tsx";
import { ComposerAttachments } from "./ComposerAttachments.tsx";
import { FlashToggle } from "./FlashToggle.tsx";
import { ContextBreakdown } from "./ContextBreakdown.tsx";
import { useChatStore, getContextView } from "../../stores/useChatStore.ts";
import { useModelsStore } from "../../stores/useModelsStore.ts";
import { useServerStore } from "../../stores/useServerStore.ts";
import { useSettingsStore } from "../../stores/useSettingsStore.ts";
import { addonSizeMb, addonState, enableAddon, useAddonProgress } from "../../chat/addons.ts";
import { prepareImage } from "../../utils/image.ts";

const MAX_IMAGES = 6;

export const Composer: React.FC<{ className?: string }> = ({ className = "" }) => {
  const [inputText, setInputText] = useState("");
  const [images, setImages] = useState<string[]>([]);
  const [isContextOpen, setIsContextOpen] = useState(false);
  const [isVisionDialogOpen, setIsVisionDialogOpen] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const pendingFiles = useRef<File[]>([]);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const activeConv = useChatStore(
    (s) => s.conversations.find((c) => c.id === s.activeConversationId) ?? null
  );
  const isGenerating = useChatStore((s) => s.isGenerating);
  const phase = useChatStore((s) => s.phase);
  const sendMessage = useChatStore((s) => s.sendMessage);
  const stopGeneration = useChatStore((s) => s.stopGeneration);
  const setConversationSettings = useChatStore((s) => s.setConversationSettings);
  const rollOver = useChatStore((s) => s.rollOver);

  const loadedModel = useModelsStore((s) => s.models.find((m) => m.dara.loaded) ?? null);
  const serverLoaded = useServerStore((s) => s.loadedModel);
  const activeModelId = loadedModel?.id ?? serverLoaded ?? undefined;
  const visionAllowed = useSettingsStore((s) => s.inference.visionEnabled);
  const defaultMaxTokens = useSettingsStore((s) => s.inference.defaultMaxTokens);
  const addon = useAddonProgress();

  const vision = addonState(loadedModel, "vision");
  const visionMb = addonSizeMb(loadedModel, "vision");
  const ctx = getContextView(activeConv);
  const contextPercent = ctx.capacity
    ? Math.min(100, Math.round((ctx.used / Math.max(1, ctx.capacity)) * 100))
    : 0;
  const locked = isGenerating || addon.busy;

  const attach = useCallback(
    async (files: File[]) => {
      for (const file of files) {
        if (images.length >= MAX_IMAGES) {
          toast.show(`Up to ${MAX_IMAGES} images per message.`, "error");
          break;
        }
        try {
          const uri = await prepareImage(file);
          setImages((prev) => (prev.length >= MAX_IMAGES ? prev : [...prev, uri]));
        } catch (err) {
          toast.show(err instanceof Error ? err.message : "Could not attach the image", "error");
        }
      }
    },
    [images.length]
  );

  /** Images can only be used when the engine has the image encoder loaded; otherwise offer to set it up first. */
  const requestImages = useCallback(
    (files: File[]) => {
      if (files.length === 0) return;
      if (!visionAllowed) {
        toast.show("Image support is turned off in Settings.", "error");
        return;
      }
      if (!loadedModel) {
        toast.show("Load a model first.", "error");
        return;
      }
      if (vision === "unavailable") {
        toast.show(
          `${loadedModel.dara.name} has no image encoder. Pick a model with image support.`,
          "error"
        );
        return;
      }
      if (vision === "active") {
        void attach(files);
        return;
      }
      pendingFiles.current = files;
      setIsVisionDialogOpen(true);
    },
    [attach, loadedModel, vision, visionAllowed]
  );

  const runVisionSetup = async () => {
    if (!loadedModel) return;
    setIsVisionDialogOpen(false);
    try {
      const cfg = await enableAddon(loadedModel.id, "vision");
      toast.show(
        cfg
          ? `Image support is on. Context is ${cfg.context.toLocaleString()} tokens.`
          : "Image support is on.",
        "success"
      );
      const files = pendingFiles.current;
      pendingFiles.current = [];
      if (files.length > 0) void attach(files);
    } catch (err) {
      pendingFiles.current = [];
      toast.show(err instanceof Error ? err.message : "Could not set up image support", "error");
    }
  };

  const handleSend = () => {
    if ((!inputText.trim() && images.length === 0) || locked) return;
    const text = inputText;
    const imgs = [...images];
    setInputText("");
    setImages([]);
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    void sendMessage(text, imgs, activeModelId);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    e.target.value = "";
    requestImages(files);
  };

  const handlePaste = (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
    const files = Array.from(e.clipboardData.files).filter((f) => f.type.startsWith("image/"));
    if (files.length > 0) {
      e.preventDefault();
      requestImages(files);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    requestImages(Array.from(e.dataTransfer.files).filter((f) => f.type.startsWith("image/")));
  };

  const autoResize = () => {
    if (!textareaRef.current) return;
    textareaRef.current.style.height = "auto";
    textareaRef.current.style.height = `${Math.min(Math.max(textareaRef.current.scrollHeight, 44), 160)}px`;
  };

  useEffect(() => {
    if (!isGenerating) textareaRef.current?.focus();
  }, [isGenerating, activeConv?.id]);

  const attachTitle =
    vision === "active"
      ? "Attach an image"
      : vision === "unavailable"
        ? "This model has no image encoder"
        : "Attach an image (sets up image support first)";

  const maxTokenCeiling = Math.max(
    512,
    Math.min(32768, ctx.capacity ? Math.floor(ctx.capacity * 0.75) : 32768)
  );
  const effectiveMax =
    activeConv?.maxTokens && activeConv.maxTokens > 0 ? activeConv.maxTokens : defaultMaxTokens;

  return (
    <div className={`p-4 bg-bg shrink-0 max-w-3xl w-full mx-auto ${className}`}>
      {addon.busy && (
        <div
          role="status"
          className="mb-2 p-3 rounded-xl border border-border bg-surface space-y-2 text-xs"
        >
          <div className="flex items-center justify-between">
            <span className="font-semibold text-text">
              {addon.kind === "vision" ? "Setting up image support" : "Setting up Flash"}
            </span>
            <span className="text-text-muted">{addon.message}</span>
          </div>
          <ProgressBar value={addon.percent} />
        </div>
      )}

      {phase === "summarizing" && (
        <div
          role="status"
          className="mb-2 px-3 py-2 rounded-xl border border-border bg-surface-raised text-xs text-text-muted flex items-center gap-2"
        >
          <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
          <span>
            The context window is nearly full. Summarizing earlier messages to start a new window
            (nothing is deleted).
          </span>
        </div>
      )}

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={`relative rounded-2xl bg-surface border transition-all shadow-sm p-3 space-y-2 ${
          isDragging
            ? "border-primary ring-2 ring-primary/30"
            : "border-border focus-within:border-primary focus-within:ring-2 focus-within:ring-primary/20"
        }`}
      >
        <ComposerAttachments
          images={images}
          onRemove={(idx) => setImages((prev) => prev.filter((_, i) => i !== idx))}
        />

        <textarea
          ref={textareaRef}
          value={inputText}
          onChange={(e) => {
            setInputText(e.target.value);
            autoResize();
          }}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
          aria-label="Message"
          placeholder="Send a message (Shift+Enter for a new line)..."
          rows={1}
          className="w-full bg-transparent p-0 text-sm text-text placeholder:text-text-subtle resize-none border-0 outline-none ring-0 focus:outline-none focus:ring-0 leading-relaxed min-h-[44px] max-h-[160px]"
          style={{ outline: "none", boxShadow: "none", border: "none" }}
        />

        <div className="flex items-center justify-between pt-2.5 mt-1 border-t border-border/40">
          <div className="flex items-center gap-2">
            <input
              type="file"
              multiple
              accept="image/png,image/jpeg,image/webp"
              ref={fileInputRef}
              onChange={handleFileSelect}
              className="hidden"
              aria-hidden="true"
              tabIndex={-1}
            />

            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={locked || vision === "unavailable"}
              className="p-1.5 rounded-lg text-text-muted hover:text-text hover:bg-surface-raised transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
              title={attachTitle}
              aria-label="Attach image"
            >
              <Icon name="image-plus" size={16} />
            </button>

            <Popover
              side="top"
              trigger={({ isOpen, toggle }) => (
                <button
                  type="button"
                  onClick={toggle}
                  aria-expanded={isOpen}
                  className="p-1.5 rounded-lg text-text-muted hover:text-text hover:bg-surface-raised transition-colors cursor-pointer"
                  title="Generation settings"
                  aria-label="Generation settings"
                >
                  <Icon name="sliders-horizontal" size={15} />
                </button>
              )}
            >
              {({ close }) => (
                <div className="space-y-4 w-72 text-left p-1">
                  <h4 className="text-xs font-bold text-text uppercase tracking-wider">
                    This conversation
                  </h4>
                  {activeConv && (
                    <>
                      <Slider
                        label="Temperature"
                        min={0}
                        max={2}
                        step={0.1}
                        value={activeConv.temperature ?? 0.7}
                        onChange={(val) =>
                          setConversationSettings(activeConv.id, { temperature: val })
                        }
                      />
                      <div className="space-y-1">
                        <Slider
                          label="Max reply length"
                          min={256}
                          max={maxTokenCeiling}
                          step={256}
                          unit=" tokens"
                          value={Math.min(maxTokenCeiling, Math.max(256, effectiveMax))}
                          onChange={(val) =>
                            setConversationSettings(activeConv.id, { maxTokens: val })
                          }
                        />
                        <p className="text-[11px] text-text-subtle leading-snug">
                          A reply stops at this length or when the context is full. If it stops
                          early you get a Continue button.
                        </p>
                      </div>
                      <Textarea
                        label="System prompt"
                        rows={3}
                        value={activeConv.systemPrompt ?? ""}
                        onChange={(e) =>
                          setConversationSettings(activeConv.id, { systemPrompt: e.target.value })
                        }
                      />
                      <Button
                        size="sm"
                        variant="secondary"
                        className="w-full"
                        disabled={locked || activeConv.messages.length < 4}
                        onClick={() => {
                          close();
                          void rollOver(activeConv.id).then((ok) => {
                            if (!ok)
                              toast.show(
                                "Not enough earlier messages to start a new window yet.",
                                "info"
                              );
                          });
                        }}
                      >
                        Start a new context window
                      </Button>
                    </>
                  )}
                  <Button size="sm" variant="primary" className="w-full" onClick={close}>
                    Done
                  </Button>
                </div>
              )}
            </Popover>

            <span className="text-border" aria-hidden="true">
              |
            </span>

            <FlashToggle />

            <span className="text-border" aria-hidden="true">
              |
            </span>

            <div className="relative">
              <button
                type="button"
                onClick={() => setIsContextOpen((prev) => !prev)}
                aria-expanded={isContextOpen}
                className="h-6 px-2.5 rounded-full border border-border bg-surface-raised hover:bg-surface-sunken transition-all flex items-center gap-1.5 text-[11px] font-mono select-none cursor-pointer"
                aria-label="Context window usage"
                title="How much of the model's context window this conversation uses"
              >
                <span
                  className={`w-1.5 h-1.5 rounded-full ${contextPercent >= 85 ? "bg-danger" : contextPercent >= 60 ? "bg-warning" : "bg-success"}`}
                />
                <span className="tabular-nums">
                  {ctx.estimated ? "~" : ""}
                  <NumberTicker value={ctx.used} /> /{" "}
                  {ctx.capacity
                    ? `${(ctx.capacity / 1024).toFixed(ctx.capacity % 1024 === 0 ? 0 : 1)}k`
                    : "-"}
                </span>
                {ctx.capacity ? (
                  <span className="text-[9px] text-text-subtle">({contextPercent}%)</span>
                ) : null}
                <Icon name="chevron-down" size={10} className="text-text-subtle" />
              </button>

              {isContextOpen && (
                <>
                  <div className="fixed inset-0 z-40" onClick={() => setIsContextOpen(false)} />
                  <div className="absolute bottom-full left-0 mb-2 z-50">
                    <ContextBreakdown
                      conversation={activeConv}
                      onClose={() => setIsContextOpen(false)}
                    />
                  </div>
                </>
              )}
            </div>
          </div>

          <div className="flex items-center gap-2">
            {isGenerating && phase === "generating" ? (
              <button
                type="button"
                onClick={stopGeneration}
                className="flex items-center gap-1.5 h-8 px-3 rounded-xl bg-danger hover:bg-danger-hover text-white text-xs font-semibold transition-all cursor-pointer shadow-sm"
              >
                <Icon name="square" size={12} />
                <span>Stop</span>
              </button>
            ) : (
              <button
                type="button"
                disabled={locked || (!inputText.trim() && images.length === 0)}
                onClick={handleSend}
                className="flex items-center gap-1.5 h-8 px-3.5 rounded-xl bg-primary hover:bg-primary-hover disabled:opacity-35 disabled:cursor-not-allowed text-white text-xs font-semibold transition-all cursor-pointer shadow-sm"
              >
                <span>Send</span>
                <Icon name="send" size={12} />
              </button>
            )}
          </div>
        </div>
      </div>

      <Dialog
        isOpen={isVisionDialogOpen}
        title="Turn on image support?"
        description={`${loadedModel?.dara.name ?? "This model"} needs its image encoder loaded to read pictures.`}
        onclose={() => {
          pendingFiles.current = [];
          setIsVisionDialogOpen(false);
        }}
        actions={
          <>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                pendingFiles.current = [];
                setIsVisionDialogOpen(false);
              }}
            >
              Not now
            </Button>
            <Button variant="primary" size="sm" onClick={() => void runVisionSetup()}>
              {vision === "install" ? "Download and turn on" : "Reload with image support"}
            </Button>
          </>
        }
      >
        <ul className="text-xs text-text-muted space-y-1.5 list-disc pl-4 py-1">
          {vision === "install" && (
            <li>Downloads the image encoder{visionMb ? ` (${Math.round(visionMb)} MB)` : ""}.</li>
          )}
          <li>Reloads the model, which takes a few seconds. Your conversations are kept.</li>
          <li>
            It uses a little GPU memory, so the context size can change slightly. The new size is
            shown afterwards.
          </li>
        </ul>
      </Dialog>
    </div>
  );
};
