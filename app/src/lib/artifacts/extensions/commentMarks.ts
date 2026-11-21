import { Extension, type Editor } from "@tiptap/react";
import { Plugin, PluginKey } from "@tiptap/pm/state";
import { Decoration, DecorationSet } from "@tiptap/pm/view";
import type { Node as PMNode } from "@tiptap/pm/model";

// Comments are anchored to a piece of text by its exact words (`quote`) and which occurrence of them it is (`nth`), not to a position, so
// the anchor survives edits elsewhere and the Markdown stays plain. If the words themselves are edited away the thread simply has no
// highlight any more ("detached") but is still listed.

export interface CommentAnchor {
  id: string;
  quote: string;
  nth: number;
}

const LEAF = "￼";
export const MAX_QUOTE = 500;
export const commentsKey = new PluginKey("commentMarks");

interface Hit {
  from: number;
  to: number;
}

/** Every place the quote appears, in document order. A quote never spans two blocks. */
export function findQuotes(doc: PMNode, quote: string): Hit[] {
  const hits: Hit[] = [];
  if (!quote) return hits;
  doc.descendants((node, pos) => {
    if (!node.isTextblock) return true;
    const text = node.textBetween(0, node.content.size, "\n", LEAF);
    let at = text.indexOf(quote);
    while (at !== -1) {
      hits.push({ from: pos + 1 + at, to: pos + 1 + at + quote.length });
      at = text.indexOf(quote, at + quote.length);
    }
    return false;
  });
  return hits;
}

export function findQuote(doc: PMNode, quote: string, nth: number): Hit | null {
  const hits = findQuotes(doc, quote);
  return hits[Math.min(Math.max(0, nth), hits.length - 1)] ?? null;
}

/** The words selected in the editor as a comment anchor (cut at the end of the block when the selection spans several). */
export function quoteForSelection(
  editor: Editor
): { quote: string; nth: number; from: number; to: number } | null {
  const { selection, doc } = editor.state;
  const $from = doc.resolve(selection.from);
  if (!$from.parent.isTextblock) return null;
  let start = selection.from;
  let end = selection.to;
  if (selection.empty) {
    // no selection: the block the cursor is in
    start = $from.start();
    end = $from.end();
  } else if (!$from.sameParent(doc.resolve(end))) end = $from.end();
  let quote = doc.textBetween(start, end, "\n", LEAF).trim();
  if (!quote) return null;
  if (quote.length > MAX_QUOTE) quote = quote.slice(0, MAX_QUOTE);
  const hits = findQuotes(doc, quote);
  const nth = Math.max(
    0,
    hits.findIndex((h) => h.from >= start)
  );
  const hit = hits[nth];
  return { quote, nth, from: hit?.from ?? start, to: hit?.to ?? end };
}

interface Options {
  getAnchors: () => CommentAnchor[];
  getActive: () => string | null;
  onSelect: (id: string) => void;
}

/** Tell the editor the anchors or the active thread changed so the highlights are redrawn. */
export function refreshCommentMarks(editor: Editor): void {
  editor.view.dispatch(editor.state.tr.setMeta(commentsKey, true));
}

export const CommentMarks = Extension.create<Options>({
  name: "commentMarks",
  addOptions() {
    return { getAnchors: () => [], getActive: () => null, onSelect: () => {} };
  },
  addProseMirrorPlugins() {
    const opts = this.options;
    return [
      new Plugin({
        key: commentsKey,
        props: {
          decorations(state) {
            const active = opts.getActive();
            const decos: Decoration[] = [];
            for (const a of opts.getAnchors()) {
              const hit = findQuote(state.doc, a.quote, a.nth);
              if (!hit) continue;
              decos.push(
                Decoration.inline(hit.from, hit.to, {
                  class: `comment-mark${a.id === active ? " comment-mark-active" : ""}`,
                  "data-comment-id": a.id
                })
              );
            }
            return DecorationSet.create(state.doc, decos);
          },
          handleClick(_view, _pos, event) {
            const el = (event.target as HTMLElement | null)?.closest?.("[data-comment-id]");
            const id = el?.getAttribute("data-comment-id");
            if (id) opts.onSelect(id);
            return false;
          }
        }
      })
    ];
  }
});
