import type { DbArtifactComment } from "../api/types.ts";

export interface Thread {
  root: DbArtifactComment;
  replies: DbArtifactComment[];
}

/** Flat comments into threads: a root and its replies in the order they were written; newest thread last. */
export function buildThreads(items: readonly DbArtifactComment[]): Thread[] {
  const roots = items
    .filter((c) => c.parent_id === null)
    .sort((a, b) => a.created_at - b.created_at);
  return roots.map((root) => ({
    root,
    replies: items
      .filter((c) => c.parent_id === root.id)
      .sort((a, b) => a.created_at - b.created_at)
  }));
}

/** Does the thread wait for an answer from Dara? True when the newest message in it was written by the user and Dara took part. */
export function awaitsDara(t: Thread): boolean {
  const all = [t.root, ...t.replies];
  const last = all[all.length - 1]!;
  return last.author === "user" && all.some((c) => c.author === "dara");
}

/** "@dara" anywhere in a comment asks Dara to act on it. */
export function mentionsDara(body: string): boolean {
  return /(^|\s)@dara\b/i.test(body);
}

/** The thread as plain text for Dara: the words it points at, then each message with who wrote it. */
export function threadForPrompt(t: Thread): string {
  const lines = [
    t.root.quote ? `Text the thread points at: "${t.root.quote}"` : "About the whole document."
  ];
  for (const c of [t.root, ...t.replies])
    lines.push(`${c.author === "dara" ? "Dara" : "User"}: ${c.body}`);
  return lines.join("\n");
}

/** What Dara is told when a comment thread needs her: the thread, and how to answer. */
export function commentInstruction(t: Thread): string {
  return [
    "A comment thread on this document needs your attention. Read it and do what the user asks or decided.",
    threadForPrompt(t),
    "Change the document only where the thread calls for it, with operations. If nothing needs to change, reply in one or two short sentences instead of operations."
  ].join("\n\n");
}
