import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import type { Editor } from "@tiptap/react";
import { Icon } from "../ui/Icon.tsx";

// Controls around the table the cursor is in, like Claude's: a grip above the column and one beside the row (each opens a small menu),
// a "+" strip on the right edge and under the bottom edge to add a column or a row, and a "..." menu for the whole table.
// Column widths are changed by dragging the border between two columns (Tiptap's column resizing).

interface Geometry {
  table: { left: number; top: number; width: number; height: number };
  col: { left: number; width: number };
  row: { top: number; height: number };
}

type MenuId = "col" | "row" | "table" | null;

interface Props {
  editor: Editor;
  /** the element the controls are positioned in (the editor's wrapper; it must be `position: relative`) */
  container: HTMLElement | null;
  disabled?: boolean;
}

const MenuItem: React.FC<{
  label: string;
  icon: string;
  danger?: boolean;
  onClick: () => void;
}> = ({ label, icon, danger, onClick }) => (
  <button
    type="button"
    role="menuitem"
    onMouseDown={(e) => e.preventDefault()}
    onClick={onClick}
    className={`flex w-full cursor-pointer items-center gap-2.5 rounded-md px-2.5 py-1.5 text-left text-xs transition-colors ${
      danger ? "text-danger hover:bg-danger-subtle" : "text-text hover:bg-primary-subtle-hover"
    }`}
  >
    <Icon name={icon} size={14} className={danger ? "" : "text-text-muted"} />
    {label}
  </button>
);

const Grip: React.FC<{
  label: string;
  className: string;
  style: React.CSSProperties;
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}> = ({ label, className, style, active, onClick, children }) => (
  <button
    type="button"
    aria-label={label}
    title={label}
    aria-haspopup="menu"
    aria-expanded={active}
    onMouseDown={(e) => e.preventDefault()}
    onClick={onClick}
    style={style}
    className={`absolute z-20 flex cursor-pointer items-center justify-center border border-border bg-surface-raised text-text-subtle shadow-xs transition-colors hover:border-border-strong hover:text-text ${className} ${
      active ? "border-border-strong text-text" : ""
    }`}
  >
    {children}
  </button>
);

export const TableControls: React.FC<Props> = ({ editor, container, disabled = false }) => {
  const [geo, setGeo] = useState<Geometry | null>(null);
  const [menu, setMenu] = useState<MenuId>(null);
  const rootRef = useRef<HTMLDivElement>(null);

  const measure = useCallback(() => {
    if (!container || disabled || !editor.isEditable || !editor.isActive("table")) {
      setGeo(null);
      return;
    }
    try {
      const { node } = editor.view.domAtPos(editor.state.selection.from);
      const el = node.nodeType === 1 ? (node as HTMLElement) : node.parentElement;
      const cell = el?.closest("td,th") as HTMLElement | null;
      const table = cell?.closest("table") as HTMLElement | null;
      const row = cell?.closest("tr") as HTMLElement | null;
      if (!cell || !table || !row) {
        setGeo(null);
        return;
      }
      const base = container.getBoundingClientRect();
      const t = table.getBoundingClientRect();
      const c = cell.getBoundingClientRect();
      const r = row.getBoundingClientRect();
      setGeo({
        table: {
          left: t.left - base.left,
          top: t.top - base.top,
          width: t.width,
          height: t.height
        },
        col: { left: c.left - base.left, width: c.width },
        row: { top: r.top - base.top, height: r.height }
      });
    } catch {
      setGeo(null);
    }
  }, [editor, container, disabled]);

  useLayoutEffect(() => {
    measure();
  }, [measure]);

  useEffect(() => {
    const on = () => requestAnimationFrame(measure);
    editor.on("selectionUpdate", on);
    editor.on("transaction", on);
    window.addEventListener("resize", on);
    const ro = container ? new ResizeObserver(on) : null;
    if (container && ro) ro.observe(container);
    return () => {
      editor.off("selectionUpdate", on);
      editor.off("transaction", on);
      window.removeEventListener("resize", on);
      ro?.disconnect();
    };
  }, [editor, container, measure]);

  useEffect(() => {
    if (!menu) return;
    const down = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setMenu(null);
    };
    const key = (e: KeyboardEvent) => e.key === "Escape" && setMenu(null);
    document.addEventListener("mousedown", down);
    document.addEventListener("keydown", key);
    return () => {
      document.removeEventListener("mousedown", down);
      document.removeEventListener("keydown", key);
    };
  }, [menu]);

  useEffect(() => {
    if (!geo) setMenu(null);
  }, [geo]);

  if (!geo) return <div ref={rootRef} aria-hidden="true" className="hidden" />;

  const run = (fn: () => boolean) => () => {
    fn();
    setMenu(null);
  };
  const chain = () => editor.chain().focus();

  /** Puts the cursor in the last cell of the row (or the first cell of the last row) so "add after" lands at the edge. */
  const atEdge = (edge: "right" | "bottom") => {
    const { node } = editor.view.domAtPos(editor.state.selection.from);
    const el = node.nodeType === 1 ? (node as HTMLElement) : node.parentElement;
    const table = el?.closest("table");
    if (!table) return;
    const cell =
      edge === "right"
        ? (el?.closest("tr")?.lastElementChild as HTMLElement | null)
        : (table.querySelector("tr:last-child > :first-child") as HTMLElement | null);
    if (!cell) return;
    const pos = editor.view.posAtDOM(cell, 0);
    chain()
      .setTextSelection(pos + 1)
      [edge === "right" ? "addColumnAfter" : "addRowAfter"]()
      .run();
  };

  const popup =
    "absolute z-30 w-48 rounded-lg border border-border bg-surface-raised p-1 shadow-lg";
  const { table, col, row } = geo;

  return (
    <div
      ref={rootRef}
      data-testid="table-controls"
      className="pointer-events-none absolute inset-0 z-20"
    >
      <div className="pointer-events-auto">
        {/* column grip, above the column */}
        <Grip
          label="Column options"
          active={menu === "col"}
          onClick={() => setMenu(menu === "col" ? null : "col")}
          className="h-4 w-9 rounded-full"
          style={{ left: col.left + col.width / 2 - 18, top: table.top - 22 }}
        >
          <Icon name="grip-horizontal" size={12} />
        </Grip>
        {menu === "col" && (
          <div
            role="menu"
            className={popup}
            style={{ left: col.left + col.width / 2 - 96, top: table.top - 16 }}
          >
            <MenuItem
              label="Insert column left"
              icon="arrow-left"
              onClick={run(() => chain().addColumnBefore().run())}
            />
            <MenuItem
              label="Insert column right"
              icon="arrow-right"
              onClick={run(() => chain().addColumnAfter().run())}
            />
            <MenuItem
              label="Delete column"
              icon="trash-2"
              danger
              onClick={run(() => chain().deleteColumn().run())}
            />
          </div>
        )}

        {/* row grip, beside the row */}
        <Grip
          label="Row options"
          active={menu === "row"}
          onClick={() => setMenu(menu === "row" ? null : "row")}
          className="h-9 w-4 rounded-full"
          style={{ left: table.left - 24, top: row.top + row.height / 2 - 18 }}
        >
          <Icon name="grip-vertical" size={12} />
        </Grip>
        {menu === "row" && (
          <div
            role="menu"
            className={popup}
            style={{ left: table.left - 18, top: row.top + row.height / 2 - 12 }}
          >
            <MenuItem
              label="Insert row above"
              icon="arrow-up"
              onClick={run(() => chain().addRowBefore().run())}
            />
            <MenuItem
              label="Insert row below"
              icon="arrow-down"
              onClick={run(() => chain().addRowAfter().run())}
            />
            <MenuItem
              label="Delete row"
              icon="trash-2"
              danger
              onClick={run(() => chain().deleteRow().run())}
            />
          </div>
        )}

        {/* add a column / a row at the edge */}
        <button
          type="button"
          aria-label="Add column"
          title="Add column"
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => atEdge("right")}
          style={{ left: table.left + table.width + 6, top: table.top, height: table.height }}
          className="absolute z-20 flex w-4 cursor-pointer items-center justify-center rounded-md border border-border bg-surface-raised text-text-subtle opacity-50 transition-opacity hover:border-border-strong hover:text-text hover:opacity-100"
        >
          <Icon name="plus" size={13} />
        </button>
        <button
          type="button"
          aria-label="Add row"
          title="Add row"
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => atEdge("bottom")}
          style={{ left: table.left, top: table.top + table.height + 6, width: table.width }}
          className="absolute z-20 flex h-4 cursor-pointer items-center justify-center rounded-md border border-border bg-surface-raised text-text-subtle opacity-50 transition-opacity hover:border-border-strong hover:text-text hover:opacity-100"
        >
          <Icon name="plus" size={13} />
        </button>

        {/* the whole table */}
        <Grip
          label="Table options"
          active={menu === "table"}
          onClick={() => setMenu(menu === "table" ? null : "table")}
          className="h-7 w-7 rounded-md"
          style={{ left: table.left + table.width + 36, top: table.top }}
        >
          <Icon name="ellipsis" size={14} />
        </Grip>
        {menu === "table" && (
          <div
            role="menu"
            className={popup}
            style={{ left: table.left + table.width - 120, top: table.top + 32 }}
          >
            <MenuItem
              label="Delete table"
              icon="trash-2"
              danger
              onClick={run(() => chain().deleteTable().run())}
            />
          </div>
        )}
      </div>
    </div>
  );
};
