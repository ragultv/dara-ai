import { useCallback, useEffect, useRef, useState } from "react";
import type { Editor } from "@tiptap/react";
import { newId } from "../api/store-api.ts";
import { useArtifactsStore } from "../stores/useArtifactsStore.ts";
import { useDocChatStore } from "../stores/useDocChatStore.ts";
import { useEffortStore } from "../stores/useEffortStore.ts";
import { getMarkdown, setMarkdown } from "./useDocEditor.ts";
import { runAssistant } from "./assistant.ts";

/** Runs the assistant against the open editor: live preview into the page, progress into the chat, one Undo per answer. */
export function useDocAssistant(editor: Editor | null, docId: string) {
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  // leaving the page stops a running answer
  useEffect(() => () => abortRef.current?.abort(), []);

  const send = useCallback(
    async (instruction: string) => {
      const text = instruction.trim();
      if (!editor || !text || abortRef.current) return;
      const chat = useDocChatStore.getState();
      const docs = useArtifactsStore.getState();
      const snapshot = getMarkdown(editor);
      const startTitle = docs.doc?.title ?? "";
      const msgId = newId();
      const controller = new AbortController();
      abortRef.current = controller;
      setBusy(true);
      chat.clearUndo(docId);
      chat.push(docId, { id: newId(), role: "user", text });
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

      editor.setEditable(false);
      let last = 0;
      try {
        const result = await runAssistant({
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
                setMarkdown(editor, md);
              }
            },
            setTitle: (t) => useArtifactsStore.getState().setTitle(t),
            steps: (steps) => patch({ steps }),
            reasoning: (reasoning, done, ms) =>
              patch({ reasoning, reasoningStreaming: !done, thinkingMs: ms })
          }
        });
        if (result.changed) {
          setMarkdown(editor, result.markdown);
          useArtifactsStore.getState().setContent(result.markdown);
          if (result.title !== (useArtifactsStore.getState().doc?.title ?? ""))
            useArtifactsStore.getState().setTitle(result.title);
        } else {
          setMarkdown(editor, snapshot);
          if ((useArtifactsStore.getState().doc?.title ?? "") !== startTitle)
            useArtifactsStore.getState().setTitle(startTitle);
        }
        patch({
          text: result.summary,
          reasoningStreaming: false,
          status: result.stopped ? "stopped" : "done",
          undo: result.changed ? { markdown: snapshot, title: startTitle } : null
        });
      } catch (err) {
        setMarkdown(editor, snapshot);
        if ((useArtifactsStore.getState().doc?.title ?? "") !== startTitle)
          useArtifactsStore.getState().setTitle(startTitle);
        patch({
          text:
            err instanceof Error && err.message ? err.message : "Dara could not edit the document.",
          reasoningStreaming: false,
          status: "error"
        });
      } finally {
        abortRef.current = null;
        editor.setEditable(true);
        setBusy(false);
      }
    },
    [editor, docId]
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);

  const undo = useCallback(
    (msgId: string) => {
      if (!editor) return;
      const chat = useDocChatStore.getState();
      const msg = (chat.byDoc[docId] ?? []).find((m) => m.id === msgId);
      if (!msg || msg.role !== "assistant" || !msg.undo) return;
      setMarkdown(editor, msg.undo.markdown);
      useArtifactsStore.getState().setContent(msg.undo.markdown);
      if ((useArtifactsStore.getState().doc?.title ?? "") !== msg.undo.title)
        useArtifactsStore.getState().setTitle(msg.undo.title);
      chat.patch(docId, msgId, { undo: null, text: `${msg.text} (undone)` });
    },
    [editor, docId]
  );

  return { busy, send, stop, undo };
}
