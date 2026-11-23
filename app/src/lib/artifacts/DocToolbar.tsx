import React, { useEffect, useReducer, useRef, useState } from "react";
import type { Editor } from "@tiptap/react";
import { Icon } from "../ui/Icon.tsx";
import { MEDIA_ACCEPT } from "./extensions/media.ts";

interface Props {
  editor: Editor;
  disabled?: boolean;
  /** a file picked with the image or video button */
  onMedia: (file: File) => void;
  /** wrap onto a second row when the column is narrower than the toolbar (a slide, beside its list and the chat) */
  wrap?: boolean;
  /** left out where there are no comments (a slide) */
  onComment?: (() => void) | undefined;
  onAsk: () => void;
}

const TEXT_STYLES: Array<{
  id: string;
  label: string;
  run: (e: Editor) => void;
  active: (e: Editor) => boolean;
}> = [
  {
    id: "p",
    label: "Text",
    run: (e) => e.chain().focus().setParagraph().run(),
    active: (e) => e.isActive("paragraph")
  },
  {
    id: "h1",
    label: "Heading 1",
    run: (e) => e.chain().focus().toggleHeading({ level: 1 }).run(),
    active: (e) => e.isActive("heading", { level: 1 })
  },
  {
    id: "h2",
    label: "Heading 2",
    run: (e) => e.chain().focus().toggleHeading({ level: 2 }).run(),
    active: (e) => e.isActive("heading", { level: 2 })
  },
  {
    id: "h3",
    label: "Heading 3",
    run: (e) => e.chain().focus().toggleHeading({ level: 3 }).run(),
    active: (e) => e.isActive("heading", { level: 3 })
  }
];

const Btn: React.FC<{
  label: string;
  icon: string;
  active?: boolean;
  disabled?: boolean;
  onClick: () => void;
}> = ({ label, icon, active, disabled, onClick }) => (
  <button
    type="button"
    aria-label={label}
    title={label}
    aria-pressed={active}
    disabled={disabled}
    // keep the selection in the editor when a toolbar button is pressed
    onMouseDown={(e) => e.preventDefault()}
    onClick={onClick}
    className={`flex h-8 w-8 cursor-pointer items-center justify-center rounded-md transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
      active
        ? "bg-surface-raised text-text"
        : "text-text-muted hover:bg-surface-raised hover:text-text"
    }`}
  >
    <Icon name={icon} size={16} />
  </button>
);

/** The formatting bar under the document: text style, emphasis, lists, table, code, quote. */
export const DocToolbar: React.FC<Props> = ({
  editor,
  disabled = false,
  onMedia,
  onComment,
  onAsk,
  wrap = false
}) => {
  const fileRef = useRef<HTMLInputElement>(null);
  const [, rerender] = useReducer((n: number) => n + 1, 0);
  const [styleOpen, setStyleOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // the active buttons follow the cursor
  useEffect(() => {
    editor.on("selectionUpdate", rerender);
    editor.on("transaction", rerender);
    return () => {
      editor.off("selectionUpdate", rerender);
      editor.off("transaction", rerender);
    };
  }, [editor]);

  useEffect(() => {
    if (!styleOpen) return;
    const down = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setStyleOpen(false);
    };
    document.addEventListener("mousedown", down);
    return () => document.removeEventListener("mousedown", down);
  }, [styleOpen]);

  const current = TEXT_STYLES.find((s) => s.active(editor)) ?? TEXT_STYLES[0]!;

  return (
    <div
      role="toolbar"
      aria-label="Formatting"
      className={`flex items-center gap-0.5 rounded-lg border border-border bg-surface-raised px-1.5 py-1 shadow-md ${
        wrap ? "max-w-full flex-wrap justify-center" : ""
      }`}
    >
      <div ref={menuRef} className="relative">
        <button
          type="button"
          disabled={disabled}
          aria-haspopup="listbox"
          aria-expanded={styleOpen}
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => setStyleOpen(!styleOpen)}
          className="flex h-8 cursor-pointer items-center gap-1 rounded-md px-2 text-xs font-semibold text-text-muted transition-colors hover:bg-surface-raised hover:text-text disabled:cursor-not-allowed disabled:opacity-40"
        >
          <span className="w-[68px] text-left">{current.label}</span>
          <Icon name="chevron-down" size={12} />
        </button>
        {styleOpen && (
          <div
            role="listbox"
            className="absolute bottom-full left-0 z-40 mb-2 w-36 rounded-lg border border-border bg-surface-raised p-1 shadow-lg"
          >
            {TEXT_STYLES.map((s) => (
              <button
                key={s.id}
                type="button"
                role="option"
                aria-selected={s.id === current.id}
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => {
                  s.run(editor);
                  setStyleOpen(false);
                }}
                className={`flex w-full cursor-pointer items-center rounded-md px-2.5 py-1.5 text-left text-xs transition-colors hover:bg-primary-subtle-hover ${
                  s.id === current.id
                    ? "bg-primary-subtle font-semibold text-text"
                    : "text-text-muted"
                }`}
              >
                {s.label}
              </button>
            ))}
          </div>
        )}
      </div>

      <span className="mx-1 h-5 w-px bg-border" aria-hidden="true" />
      <Btn
        label="Bold"
        icon="bold"
        disabled={disabled}
        active={editor.isActive("bold")}
        onClick={() => editor.chain().focus().toggleBold().run()}
      />
      <Btn
        label="Italic"
        icon="italic"
        disabled={disabled}
        active={editor.isActive("italic")}
        onClick={() => editor.chain().focus().toggleItalic().run()}
      />
      <Btn
        label="Inline code"
        icon="code"
        disabled={disabled}
        active={editor.isActive("code")}
        onClick={() => editor.chain().focus().toggleCode().run()}
      />
      <span className="mx-1 h-5 w-px bg-border" aria-hidden="true" />
      <Btn
        label="Bulleted list"
        icon="list"
        disabled={disabled}
        active={editor.isActive("bulletList")}
        onClick={() => editor.chain().focus().toggleBulletList().run()}
      />
      <Btn
        label="Numbered list"
        icon="list-ordered"
        disabled={disabled}
        active={editor.isActive("orderedList")}
        onClick={() => editor.chain().focus().toggleOrderedList().run()}
      />
      <Btn
        label="Checklist"
        icon="list-checks"
        disabled={disabled}
        active={editor.isActive("taskList")}
        onClick={() => editor.chain().focus().toggleTaskList().run()}
      />
      <span className="mx-1 h-5 w-px bg-border" aria-hidden="true" />
      <Btn
        label="Table"
        icon="table"
        disabled={disabled}
        active={editor.isActive("table")}
        onClick={() =>
          editor.isActive("table")
            ? editor.chain().focus().deleteTable().run()
            : editor.chain().focus().insertTable({ rows: 3, cols: 3, withHeaderRow: true }).run()
        }
      />
      <Btn
        label="Image or video"
        icon="image"
        disabled={disabled}
        onClick={() => fileRef.current?.click()}
      />
      <input
        ref={fileRef}
        type="file"
        accept={MEDIA_ACCEPT}
        hidden
        aria-hidden="true"
        tabIndex={-1}
        onChange={(e) => {
          const f = e.target.files?.[0];
          e.target.value = "";
          if (f) onMedia(f);
        }}
      />
      <Btn
        label="Code block"
        icon="square-code"
        disabled={disabled}
        active={editor.isActive("codeBlock")}
        onClick={() => editor.chain().focus().toggleCodeBlock().run()}
      />
      <Btn
        label="Quote"
        icon="quote"
        disabled={disabled}
        active={editor.isActive("blockquote")}
        onClick={() => editor.chain().focus().toggleBlockquote().run()}
      />
      {/* a slide has no divider: that line is what ends a slide */}
      {editor.extensionManager.extensions.some((x) => x.name === "horizontalRule") && (
        <Btn
          label="Divider"
          icon="minus"
          disabled={disabled}
          onClick={() => editor.chain().focus().setHorizontalRule().run()}
        />
      )}
      <span className="mx-1 h-5 w-px bg-border" aria-hidden="true" />
      {onComment && (
        <Btn label="Comment" icon="message-circle" disabled={disabled} onClick={onComment} />
      )}
      <button
        type="button"
        onMouseDown={(e) => e.preventDefault()}
        onClick={onAsk}
        className="ml-0.5 flex h-8 cursor-pointer items-center gap-1.5 rounded-md bg-surface-sunken px-3 text-xs font-semibold text-text transition-colors hover:bg-primary-subtle-hover"
      >
        <Icon name="sparkles" size={13} />
        Ask Dara
      </button>
    </div>
  );
};
