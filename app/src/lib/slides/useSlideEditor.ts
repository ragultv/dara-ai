import { useEffect, useRef } from "react";
import { useEditor, type Editor } from "@tiptap/react";
import StarterKit from "@tiptap/starter-kit";
import Placeholder from "@tiptap/extension-placeholder";
import Link from "@tiptap/extension-link";
import Table from "@tiptap/extension-table";
import TableRow from "@tiptap/extension-table-row";
import TableHeader from "@tiptap/extension-table-header";
import TableCell from "@tiptap/extension-table-cell";
import TaskList from "@tiptap/extension-task-list";
import TaskItem from "@tiptap/extension-task-item";
import { Markdown } from "tiptap-markdown";
import { DaraMedia, insertMedia, isMediaFile } from "../artifacts/extensions/media.ts";
import { getMarkdown } from "../artifacts/useDocEditor.ts";

interface Options {
  /** the slide's Markdown when the editor is made (one editor per slide, so each keeps its own undo history) */
  initial: string;
  onChange: (markdown: string) => void;
  /** the deck the slide's images are stored with */
  docId: string;
  onMediaProblem: (message: string) => void;
}

/**
 * The editor of one slide: the same ProseMirror and Markdown as a document, but a slide cannot hold a horizontal rule (that line is what
 * ends a slide), and there are no sections, widgets or comments. Raw HTML is never rendered and images load only from the deck's own
 * storage, exactly as in a document.
 */
export function useSlideEditor(options: Options): Editor | null {
  const latest = useRef(options);
  latest.current = options;
  const editorRef = useRef<Editor | null>(null);

  const upload = (files: File[]) => {
    const ed = editorRef.current;
    if (!ed) return;
    void (async () => {
      for (const f of files) {
        const problem = await insertMedia(ed, latest.current.docId, f);
        if (problem) latest.current.onMediaProblem(problem);
      }
    })();
  };

  const editor = useEditor({
    extensions: [
      StarterKit.configure({ heading: { levels: [1, 2, 3] }, horizontalRule: false }),
      Placeholder.configure({ placeholder: "Click to write, or ask Dara" }),
      Link.configure({ openOnClick: false, autolink: true }),
      Table.configure({ resizable: false }),
      TableRow,
      TableHeader,
      TableCell,
      TaskList,
      TaskItem.configure({ nested: true }),
      DaraMedia.configure({ getDocId: () => latest.current.docId }),
      Markdown.configure({
        html: false,
        tightLists: true,
        bulletListMarker: "-",
        linkify: false,
        breaks: false
      })
    ],
    content: options.initial,
    editorProps: {
      // not the document's typography class: a slide is coloured by its theme, not by the app's
      attributes: { class: "slide-editor", spellcheck: "true" },
      handlePaste: (_view, event) => {
        const files = Array.from(event.clipboardData?.files ?? []).filter(isMediaFile);
        if (!files.length) return false;
        upload(files);
        return true;
      },
      handleDrop: (_view, event) => {
        const files = Array.from(event.dataTransfer?.files ?? []).filter(isMediaFile);
        if (!files.length) return false;
        event.preventDefault();
        upload(files);
        return true;
      }
    },
    onUpdate: ({ editor: e }) => latest.current.onChange(getMarkdown(e))
  });

  useEffect(() => {
    editorRef.current = editor;
  }, [editor]);

  return editor;
}
