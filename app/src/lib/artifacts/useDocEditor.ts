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
}

/**
 * The document editor: ProseMirror through Tiptap, with Markdown in and out. Raw HTML in the Markdown is never rendered (html: false), and links
 * keep Tiptap's protocol check, so a document written by the model cannot inject anything.
 */
export function useDocEditor({ initial, onChange }: Options): Editor | null {
  return useEditor({
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
      Markdown.configure({
        html: false,
        tightLists: true,
        bulletListMarker: "-",
        linkify: false,
        breaks: false
      })
    ],
    content: initial,
    editorProps: {
      attributes: { class: "doc-editor oprel-markdown", spellcheck: "true" }
    },
    onUpdate: ({ editor }) => onChange(getMarkdown(editor))
  });
}
