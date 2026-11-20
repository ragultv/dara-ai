import type { Editor } from "@tiptap/react";

// PDF export: the page is printed with its own look. A copy of the editor's content (with the editing aids removed: fold chevrons,
// comment highlights, table grips, widgets) is placed in #print-root, the print stylesheet in app.css hides the rest of the app and
// shows only that copy in a light palette, and the system print dialog saves it as a PDF ("Save as PDF" / "Microsoft Print to PDF").

const ROOT_ID = "print-root";

/** The editor's content without anything that only helps while editing. */
export function cleanCopy(source: HTMLElement): HTMLElement {
  const copy = source.cloneNode(true) as HTMLElement;
  copy.removeAttribute("contenteditable");
  copy.removeAttribute("tabindex");
  copy
    .querySelectorAll(
      ".fold-toggle, .column-resize-handle, .ProseMirror-gapcursor, .ProseMirror-trailingBreak"
    )
    .forEach((e) => e.remove());
  copy.querySelectorAll(".doc-widget").forEach((w) => {
    const note = document.createElement("p");
    note.className = "print-note";
    note.textContent = "Interactive widget: open this document in Dara to use it.";
    w.replaceWith(note);
  });
  // the code of a hidden widget is not part of the page
  copy.querySelectorAll("pre.widget-code").forEach((e) => e.remove());
  copy.querySelectorAll(".section-folded").forEach((e) => e.classList.remove("section-folded"));
  copy.querySelectorAll(".comment-mark, .comment-mark-active").forEach((e) => {
    e.classList.remove("comment-mark", "comment-mark-active");
    e.removeAttribute("data-comment-id");
  });
  copy.querySelectorAll("[contenteditable]").forEach((e) => e.removeAttribute("contenteditable"));
  copy.querySelectorAll("video").forEach((v) => {
    const note = document.createElement("p");
    note.className = "print-note";
    note.textContent = "Video: open this document in Dara to play it.";
    v.closest("figure")?.replaceWith(note);
  });
  return copy;
}

/** Opens the print dialog for the open document. Resolves once the dialog has closed. */
export function printDocument(editor: Editor, title: string): Promise<void> {
  document.getElementById(ROOT_ID)?.remove();
  const root = document.createElement("div");
  root.id = ROOT_ID;
  if (title.trim()) {
    const h = document.createElement("h1");
    h.className = "print-title";
    h.textContent = title.trim();
    root.append(h);
  }
  root.append(cleanCopy(editor.view.dom));
  document.body.append(root);
  const previousTitle = document.title;
  document.title = title.trim() || "Untitled"; // the PDF's suggested file name

  return new Promise((resolve) => {
    const done = () => {
      window.removeEventListener("afterprint", done);
      document.title = previousTitle;
      root.remove();
      resolve();
    };
    window.addEventListener("afterprint", done);
    // images are already loaded (blob URLs); give the layout one frame, then print
    requestAnimationFrame(() => {
      window.print();
      // a webview that never raises afterprint: clean up much later (the preview has long since been made)
      setTimeout(done, 120_000);
    });
  });
}
