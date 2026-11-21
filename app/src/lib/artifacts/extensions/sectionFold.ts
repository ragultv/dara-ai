import { Extension } from "@tiptap/react";
import { Plugin, PluginKey } from "@tiptap/pm/state";
import { Decoration, DecorationSet } from "@tiptap/pm/view";

// Collapsible sections, like Claude's: a chevron before every heading folds everything under it (up to the next heading of the same or a
// higher level). Only the view changes; the document, the Markdown and what Dara reads are untouched. Folds are remembered by the
// heading's order in the page for as long as the page is open.

interface FoldState {
  folded: Set<number>;
}
export const sectionFoldKey = new PluginKey<FoldState>("sectionFold");

const CHEVRON =
  '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m9 18 6-6-6-6"/></svg>';

export const SectionFold = Extension.create({
  name: "sectionFold",
  addProseMirrorPlugins() {
    return [
      new Plugin<FoldState>({
        key: sectionFoldKey,
        state: {
          init: () => ({ folded: new Set<number>() }),
          apply(tr, value) {
            const toggle = tr.getMeta(sectionFoldKey) as { toggle: number } | undefined;
            if (!toggle) return value;
            const next = new Set(value.folded);
            if (next.has(toggle.toggle)) next.delete(toggle.toggle);
            else next.add(toggle.toggle);
            return { folded: next };
          }
        },
        props: {
          decorations(state) {
            const { folded } = sectionFoldKey.getState(state)!;
            const decos: Decoration[] = [];
            let index = -1;
            let foldLevel: number | null = null; // level of the heading whose section is being hidden
            state.doc.forEach((node, offset) => {
              const isHeading = node.type.name === "heading";
              const level = isHeading ? (node.attrs.level as number) : 0;
              if (isHeading && foldLevel !== null && level <= foldLevel) foldLevel = null;
              if (foldLevel !== null) {
                decos.push(
                  Decoration.node(offset, offset + node.nodeSize, { class: "section-folded" })
                );
                if (isHeading) index++; // keep the numbering of hidden headings stable
                return;
              }
              if (!isHeading) return;
              index++;
              const i = index;
              const isFolded = folded.has(i);
              if (isFolded) foldLevel = level;
              decos.push(
                Decoration.widget(
                  offset + 1,
                  (view) => {
                    const b = document.createElement("button");
                    b.type = "button";
                    b.className = "fold-toggle";
                    b.contentEditable = "false";
                    b.dataset.folded = String(isFolded);
                    b.setAttribute("aria-expanded", String(!isFolded));
                    b.setAttribute("aria-label", isFolded ? "Expand section" : "Collapse section");
                    b.title = isFolded ? "Expand section" : "Collapse section";
                    b.innerHTML = CHEVRON;
                    b.addEventListener("mousedown", (e) => e.preventDefault());
                    b.addEventListener("click", () => {
                      view.dispatch(view.state.tr.setMeta(sectionFoldKey, { toggle: i }));
                    });
                    return b;
                  },
                  { side: -1, key: `fold-${i}-${isFolded ? 1 : 0}`, ignoreSelection: true }
                )
              );
            });
            return DecorationSet.create(state.doc, decos);
          }
        }
      })
    ];
  }
});
