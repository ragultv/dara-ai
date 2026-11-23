import { useCallback, useEffect, useRef, useState } from "react";
import type { Editor } from "@tiptap/react";
import type { ArtifactKind } from "../api/types.ts";
import { newId } from "../api/store-api.ts";
import { useArtifactsStore } from "../stores/useArtifactsStore.ts";
import { useDocChatStore } from "../stores/useDocChatStore.ts";
import { useEffortStore } from "../stores/useEffortStore.ts";
import { getMarkdown, setMarkdown } from "./useDocEditor.ts";
import { useCommentsStore } from "../stores/useCommentsStore.ts";
import { runAssistant, type AssistantResult } from "./assistant.ts";

/**
 * What the assistant writes into: the text it reads, a place for the live preview and the final text, and a switch that stops the person
 * typing while it works. A document's target is its editor; a deck's is the deck page.
 */
export interface DocTarget {
  get: () => string;
  set: (markdown: string) => void;
  setEditable: (editable: boolean) => void;
}

export function editorTarget(editor: Editor): DocTarget {
  return {
    get: () => getMarkdown(editor),
    set: (md) => setMarkdown(editor, md),
    setEditable: (e) => editor.setEditable(e)
  };
}

export interface SendOptions {
  /** what the chat shows as the user's message, when the instruction is longer than what they typed (a comment reply) */
  display?: string;
  /** called with the outcome once the task has finished without an error */
  onResult?: (result: AssistantResult) => void | Promise<void>;
}

/** Runs the assistant against the open editor: live preview into the page, progress into the chat, one Undo per answer. */
export function useDocAssistant(
  target: DocTarget | null,
  docId: string,
  kind: Extract<ArtifactKind, "doc" | "slides"> = "doc"
) {
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  // leaving the page stops a running answer
  useEffect(() => () => abortRef.current?.abort(), []);

  const send = useCallback(
    async (instruction: string, opts: SendOptions = {}) => {
      const text = instruction.trim();
      if (!target || !text || abortRef.current) return;
      const chat = useDocChatStore.getState();
      const docs = useArtifactsStore.getState();
      await docs.flush(); // the saved text is what a "Before Dara" version will keep
      const snapshot = target.get();
      const startTitle = docs.doc?.title ?? "";
      const msgId = newId();
      const controller = new AbortController();
      abortRef.current = controller;
      setBusy(true);
      chat.clearUndo(docId);
      const userMsg = { id: newId(), role: "user" as const, text: opts.display ?? text };
      chat.push(docId, userMsg);
      void chat.persist(docId, userMsg);
      chat.push(docId, {
        id: msgId,
        role: "assistant",
        text: "",
        steps: [],
        reasoning: "",
        reasoningStreaming: false,
        thinkingMs: 0,
        status: "running",
        undo: null
      });
      const patch = (p: Parameters<typeof chat.patch>[2]) =>
        useDocChatStore.getState().patch(docId, msgId, p);
      /** saves the answer with the document once it has an outcome (done, stopped or failed) */
      const save = () => {
        const m = (useDocChatStore.getState().byDoc[docId] ?? []).find((x) => x.id === msgId);
        if (m) void useDocChatStore.getState().persist(docId, m);
      };

      target.setEditable(false);
      let last = 0;
      const startedAt = Date.now();
      const began: Record<number, string> = {}; // when each step first became active, as m:ss into the task
      const clock = () => {
        const sec = Math.floor((Date.now() - startedAt) / 1000);
        return `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, "0")}`;
      };
      try {
        const result = await runAssistant({
          kind,
          markdown: snapshot,
          title: startTitle,
          instruction: text,
          effort: useEffortStore.getState().effort,
          signal: controller.signal,
          hooks: {
            setDoc: (md) => {
              const now = Date.now();
              if (now - last > 150) {
                last = now;
                target.set(md);
              }
            },
            setTitle: (t) => useArtifactsStore.getState().setTitle(t),
            steps: (steps) =>
              patch({
                steps: steps.map((st, i) => {
                  if (st.state !== "todo" && began[i] === undefined) began[i] = clock();
                  return { ...st, time: began[i] ?? "" };
                })
              }),
            say: (t) => patch({ text: t }),
            reasoning: (reasoning, done, ms) =>
              patch({ reasoning, reasoningStreaming: !done, thinkingMs: ms })
          }
        });
        if (result.changed) {
          target.set(result.markdown);
          useArtifactsStore.getState().setContent(result.markdown, {
            snapshot: `Before Dara: ${(opts.display ?? text).replace(/\s+/g, " ").slice(0, 70)}`
          });
          if (result.title !== (useArtifactsStore.getState().doc?.title ?? ""))
            useArtifactsStore.getState().setTitle(result.title);
        } else {
          target.set(snapshot);
          if ((useArtifactsStore.getState().doc?.title ?? "") !== startTitle)
            useArtifactsStore.getState().setTitle(startTitle);
        }
        // questions Dara needs answered go to the comments, next to the words they are about
        for (const q of result.questions)
          await useCommentsStore.getState().add({
            author: "dara",
            body: q.body,
            quote: q.quote,
            nth: q.nth
          });
        patch({
          text: result.reply ?? result.summary,
          reasoningStreaming: false,
          status: result.stopped ? "stopped" : "done",
          undo: result.changed ? { markdown: snapshot, title: startTitle } : null
        });
        save();
        await opts.onResult?.(result);
      } catch (err) {
        target.set(snapshot);
        if ((useArtifactsStore.getState().doc?.title ?? "") !== startTitle)
          useArtifactsStore.getState().setTitle(startTitle);
        patch({
          text:
            err instanceof Error && err.message ? err.message : "Dara could not edit the document.",
          reasoningStreaming: false,
          status: "error"
        });
        save();
      } finally {
        abortRef.current = null;
        target.setEditable(true);
        setBusy(false);
      }
    },
    [target, docId, kind]
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);

  const undo = useCallback(
    (msgId: string) => {
      if (!target) return;
      const chat = useDocChatStore.getState();
      const msg = (chat.byDoc[docId] ?? []).find((m) => m.id === msgId);
      if (!msg || msg.role !== "assistant" || !msg.undo) return;
      target.set(msg.undo.markdown);
      useArtifactsStore.getState().setContent(msg.undo.markdown);
      if ((useArtifactsStore.getState().doc?.title ?? "") !== msg.undo.title)
        useArtifactsStore.getState().setTitle(msg.undo.title);
      chat.patch(docId, msgId, { undo: null, text: `${msg.text} (undone)` });
    },
    [target, docId]
  );

  return { busy, send, stop, undo };
}
