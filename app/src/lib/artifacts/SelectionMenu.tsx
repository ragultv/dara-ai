import React, { useEffect, useRef, useState } from "react";
import { BubbleMenu, type Editor } from "@tiptap/react";
import { Icon } from "../ui/Icon.tsx";

// The small toolbar that appears over selected text, like Claude's: text style, bold, italic, strikethrough, code, list and link.

const STYLES: Array<{ label: string; level: 0 | 1 | 2 | 3 }> = [
  { label: "Text", level: 0 },
  { label: "Heading 1", level: 1 },
  { label: "Heading 2", level: 2 },
  { label: "Heading 3", level: 3 }
];

/** "example.com" becomes https://example.com; only web and mail links are accepted. */
export function normalizeLink(input: string): string | null {
  const v = input.trim();
  if (!v) return null;
  const withScheme = /^[a-z][a-z0-9+.-]*:/i.test(v) ? v : `https://${v}`;
  return /^(https?:|mailto:)/i.test(withScheme) ? withScheme : null;
}

const Btn: React.FC<{
  label: string;
  icon: string;
  active?: boolean;
  onClick: () => void;
}> = ({ label, icon, active, onClick }) => (
  <button
    type="button"
    aria-label={label}
    title={label}
    aria-pressed={active}
    onMouseDown={(e) => e.preventDefault()}
    onClick={onClick}
    className={`flex h-7 w-7 cursor-pointer items-center justify-center rounded-md transition-colors ${
      active
        ? "bg-primary-subtle text-text"
        : "text-text-muted hover:bg-primary-subtle-hover hover:text-text"
    }`}
  >
    <Icon name={icon} size={14} />
  </button>
);

export const SelectionMenu: React.FC<{ editor: Editor; onComment: () => void }> = ({
  editor,
  onComment
}) => {
  const [, tick] = useState(0);
  const [styleOpen, setStyleOpen] = useState(false);
  const [linkOpen, setLinkOpen] = useState(false);
  const [href, setHref] = useState("");
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const on = () => tick((n) => n + 1);
    editor.on("selectionUpdate", on);
    editor.on("transaction", on);
    return () => {
      editor.off("selectionUpdate", on);
      editor.off("transaction", on);
    };
  }, [editor]);

  const chain = () => editor.chain().focus();
  const current = STYLES.find((s) =>
    s.level === 0 ? editor.isActive("paragraph") : editor.isActive("heading", { level: s.level })
  );

  const openLink = () => {
    setHref((editor.getAttributes("link").href as string | undefined) ?? "");
    setLinkOpen(true);
    setStyleOpen(false);
  };
  const applyLink = () => {
    const url = normalizeLink(href);
    if (url) chain().extendMarkRange("link").setLink({ href: url }).run();
    setLinkOpen(false);
  };

  return (
    <BubbleMenu
      editor={editor}
      tippyOptions={{
        duration: 120,
        placement: "top",
        maxWidth: "none",
        onHidden: () => {
          setStyleOpen(false);
          setLinkOpen(false);
        }
      }}
      shouldShow={({ editor: e, from, to }) =>
        e.isEditable && from !== to && !e.isActive("codeBlock") && !e.isActive("table")
      }
    >
      <div
        ref={rootRef}
        role="toolbar"
        aria-label="Format selection"
        className="relative flex items-center gap-0.5 rounded-lg border border-border bg-surface-raised px-1.5 py-1 shadow-lg"
      >
        {linkOpen ? (
          <form
            className="flex items-center gap-1.5 px-1"
            onSubmit={(e) => {
              e.preventDefault();
              applyLink();
            }}
          >
            <input
              autoFocus
              value={href}
              onChange={(e) => setHref(e.target.value)}
              onKeyDown={(e) => e.key === "Escape" && setLinkOpen(false)}
              placeholder="Paste a link"
              aria-label="Link address"
              className="h-7 w-56 rounded-md border border-border bg-bg px-2 text-xs text-text outline-none placeholder:text-text-subtle focus:border-border-strong"
            />
            <button
              type="submit"
              className="h-7 cursor-pointer rounded-md bg-primary px-2.5 text-xs font-semibold text-white transition-colors hover:bg-primary-hover"
            >
              Apply
            </button>
            {editor.isActive("link") && (
              <button
                type="button"
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => {
                  chain().extendMarkRange("link").unsetLink().run();
                  setLinkOpen(false);
                }}
                className="h-7 cursor-pointer rounded-md px-2 text-xs text-text-muted hover:bg-primary-subtle-hover hover:text-text"
              >
                Remove
              </button>
            )}
          </form>
        ) : (
          <>
            <Btn label="Comment" icon="message-circle" onClick={onComment} />
            <span className="mx-1 h-5 w-px bg-border" aria-hidden="true" />
            <div className="relative">
              <button
                type="button"
                aria-haspopup="listbox"
                aria-expanded={styleOpen}
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => setStyleOpen(!styleOpen)}
                className="flex h-7 cursor-pointer items-center gap-1 rounded-md px-2 text-xs font-semibold text-text-muted transition-colors hover:bg-primary-subtle-hover hover:text-text"
              >
                <Icon name="type" size={14} />
                <Icon name="chevron-down" size={11} />
              </button>
              {styleOpen && (
                <div
                  role="listbox"
                  className="absolute left-0 top-full z-40 mt-2 w-36 rounded-lg border border-border bg-surface-raised p-1 shadow-lg"
                >
                  {STYLES.map((s) => (
                    <button
                      key={s.label}
                      type="button"
                      role="option"
                      aria-selected={current?.level === s.level}
                      onMouseDown={(e) => e.preventDefault()}
                      onClick={() => {
                        if (s.level === 0) chain().setParagraph().run();
                        else chain().toggleHeading({ level: s.level }).run();
                        setStyleOpen(false);
                      }}
                      className={`flex w-full cursor-pointer items-center rounded-md px-2.5 py-1.5 text-left text-xs transition-colors hover:bg-primary-subtle-hover ${
                        current?.level === s.level
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
              active={editor.isActive("bold")}
              onClick={() => chain().toggleBold().run()}
            />
            <Btn
              label="Italic"
              icon="italic"
              active={editor.isActive("italic")}
              onClick={() => chain().toggleItalic().run()}
            />
            <Btn
              label="Strikethrough"
              icon="strikethrough"
              active={editor.isActive("strike")}
              onClick={() => chain().toggleStrike().run()}
            />
            <Btn
              label="Inline code"
              icon="code"
              active={editor.isActive("code")}
              onClick={() => chain().toggleCode().run()}
            />
            <span className="mx-1 h-5 w-px bg-border" aria-hidden="true" />
            <Btn
              label="Bulleted list"
              icon="list"
              active={editor.isActive("bulletList")}
              onClick={() => chain().toggleBulletList().run()}
            />
            <Btn label="Link" icon="link" active={editor.isActive("link")} onClick={openLink} />
          </>
        )}
      </div>
    </BubbleMenu>
  );
};
