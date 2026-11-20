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
import { SectionFold } from "./extensions/sectionFold.ts";
import { CommentMarks, type CommentAnchor } from "./extensions/commentMarks.ts";
import { DaraMedia, insertMedia, isMediaFile } from "./extensions/media.ts";
import { Widgets } from "./extensions/widgets.ts";

/** The document as Markdown (the stored format). */
export function getMarkdown(editor: Editor): string {
  const storage = editor.storage as unknown as { markdown?: { getMarkdown: () => string } };
  return storage.markdown?.getMarkdown() ?? "";
}

/** Replaces the whole document from Markdown without raising an update (the caller saves it). */
export function setMarkdown(editor: Editor, markdown: string): void {
  editor.commands.setContent(markdown, false);
}

interface Options {
  initial: string;
  onChange: (markdown: string) => void;
  /** the document being edited: images and video are stored with it */
  docId: string;
  /** the comment threads shown as highlights in the page */
  comments: {
    getAnchors: () => CommentAnchor[];
    getActive: () => string | null;
    onSelect: (id: string) => void;
  };
  /** something went wrong adding a pasted or dropped file */
  onMediaProblem: (message: string) => void;
}

/**
 * The document editor: ProseMirror through Tiptap, with Markdown in and out. Raw HTML in the Markdown is never rendered (html: false),
 * links keep Tiptap's protocol check and images load only from this document's own storage, so a document written by the model cannot
 * inject anything or contact a server. Besides the usual blocks it has collapsible sections, comment highlights, images and video, and
 * live widgets (a code block whose language is "widget").
 */
export function useDocEditor(options: Options): Editor | null {
  // the extensions are built once, so they read the latest callbacks through this ref
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
      StarterKit.configure({ heading: { levels: [1, 2, 3] } }),
      Placeholder.configure({ placeholder: "Start typing, or ask Dara to write" }),
      Link.configure({ openOnClick: false, autolink: true }),
      Table.configure({ resizable: true, cellMinWidth: 80, lastColumnResizable: true }),
      TableRow,
      TableHeader,
      TableCell,
      TaskList,
      TaskItem.configure({ nested: true }),
      DaraMedia.configure({ getDocId: () => latest.current.docId }),
      SectionFold,
      Widgets,
      CommentMarks.configure({
        getAnchors: () => latest.current.comments.getAnchors(),
        getActive: () => latest.current.comments.getActive(),
        onSelect: (id) => latest.current.comments.onSelect(id)
      }),
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
      attributes: { class: "doc-editor oprel-markdown", spellcheck: "true" },
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
