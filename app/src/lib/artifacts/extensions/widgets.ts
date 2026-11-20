import { Extension } from "@tiptap/react";
import { Plugin, PluginKey } from "@tiptap/pm/state";
import { Decoration, DecorationSet } from "@tiptap/pm/view";

// Widgets: a fenced code block whose language is `widget` is a small interactive piece (a calculator, a chart, a checklist...) written as
// one self-contained HTML page with its own CSS and JavaScript. It is shown live under its code, in a sandboxed frame: scripts run, but
// the frame has no network, no storage, no access to this app and cannot navigate away or open windows.

interface WidgetState {
  shown: Set<number>; // widgets whose code is open
}
export const widgetsKey = new PluginKey<WidgetState>("widgets");

export const WIDGET_LANGUAGE = "widget";
const MIN_HEIGHT = 48;
const MAX_HEIGHT = 640;

/** The page the frame loads. The Content-Security-Policy blocks every outside request; only inline script and style run. */
export function widgetSrcDoc(
  code: string,
  colors: Record<string, string>,
  frameId: string
): string {
  const vars = Object.entries(colors)
    .map(([k, v]) => `--${k}:${v.replace(/[<>;{}]/g, "")}`)
    .join(";");
  const csp =
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; media-src data:; font-src data:";
  return [
    '<!doctype html><html><head><meta charset="utf-8">',
    `<meta http-equiv="Content-Security-Policy" content="${csp}">`,
    `<style>:root{${vars};color-scheme:${colors["scheme"] ?? "light"}}html,body{margin:0}body{padding:12px;font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--text);background:transparent}button,input,select,textarea{font:inherit;color:inherit}</style>`,
    "</head><body>",
    code,
    `<script>(function(){var send=function(){parent.postMessage({daraWidgetHeight:document.documentElement.scrollHeight,frame:${JSON.stringify(frameId)}},"*")};new ResizeObserver(send).observe(document.documentElement);window.addEventListener("load",send);send();})();</script>`,
    "</body></html>"
  ].join("");
}

const frames = new Map<string, HTMLIFrameElement>();
let listening = false;
function listen(): void {
  if (listening) return;
  listening = true;
  window.addEventListener("message", (e: MessageEvent) => {
    const data = e.data as { daraWidgetHeight?: unknown; frame?: unknown } | null;
    if (!data || typeof data.daraWidgetHeight !== "number" || typeof data.frame !== "string")
      return;
    const frame = frames.get(data.frame);
    if (!frame || e.source !== frame.contentWindow) return; // only the frame that owns the id may resize it
    frame.style.height = `${Math.min(MAX_HEIGHT, Math.max(MIN_HEIGHT, Math.ceil(data.daraWidgetHeight)))}px`;
  });
}

function themeColors(): Record<string, string> {
  const css = getComputedStyle(document.documentElement);
  const get = (n: string) => css.getPropertyValue(`--color-${n}`).trim();
  return {
    text: get("text") || "#111",
    muted: get("text-muted") || "#555",
    bg: get("bg") || "#fff",
    surface: get("surface-raised") || "#f5f5f5",
    border: get("border") || "#ddd",
    primary: get("primary") || "#6d28d9",
    scheme: document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light"
  };
}

let counter = 0;
function buildWidget(code: string, index: number, toggle: () => void): HTMLElement {
  listen();
  const root = document.createElement("div");
  root.className = "doc-widget";
  root.contentEditable = "false";
  const bar = document.createElement("div");
  bar.className = "doc-widget-bar";
  const label = document.createElement("span");
  label.textContent = "Widget";
  const btn = document.createElement("button");
  btn.type = "button";
  // the label follows the code block's state through CSS (.widget-code-open + .doc-widget), so the frame is not rebuilt when it flips
  btn.innerHTML = '<span class="wl-show">Show code</span><span class="wl-hide">Hide code</span>';
  btn.addEventListener("mousedown", (e) => e.preventDefault());
  btn.addEventListener("click", toggle);
  bar.append(label, btn);
  const frame = document.createElement("iframe");
  const id = `w${++counter}-${index}`;
  frame.setAttribute("sandbox", "allow-scripts"); // no allow-same-origin: the widget cannot reach this app's storage or cookies
  frame.setAttribute("referrerpolicy", "no-referrer");
  frame.setAttribute("title", `Widget ${index + 1}`);
  frame.className = "doc-widget-frame";
  frame.style.height = "120px";
  frames.set(id, frame);
  // a data: page, not srcdoc: a srcdoc frame inherits this app's Content-Security-Policy (no inline script), a data: page has its own
  frame.src = `data:text/html;charset=utf-8,${encodeURIComponent(widgetSrcDoc(code, themeColors(), id))}`;
  root.append(bar, frame);
  return root;
}

export const Widgets = Extension.create({
  name: "widgets",
  addProseMirrorPlugins() {
    return [
      new Plugin<WidgetState>({
        key: widgetsKey,
        state: {
          init: () => ({ shown: new Set<number>() }),
          apply(tr, value) {
            const t = tr.getMeta(widgetsKey) as { toggle: number } | undefined;
            if (!t) return value;
            const next = new Set(value.shown);
            if (next.has(t.toggle)) next.delete(t.toggle);
            else next.add(t.toggle);
            return { shown: next };
          }
        },
        props: {
          decorations(state) {
            const { shown } = widgetsKey.getState(state)!;
            const decos: Decoration[] = [];
            let index = -1;
            state.doc.forEach((node, offset) => {
              if (node.type.name !== "codeBlock" || node.attrs.language !== WIDGET_LANGUAGE) return;
              index++;
              const i = index;
              const code = node.textContent;
              // the code is open when asked for, and always while the cursor is inside it (an empty widget has to be typed into)
              const sel = state.selection;
              const inside = sel.from > offset && sel.to < offset + node.nodeSize;
              const open = shown.has(i) || inside;
              decos.push(
                Decoration.node(offset, offset + node.nodeSize, {
                  class: open ? "widget-code widget-code-open" : "widget-code"
                })
              );
              decos.push(
                Decoration.widget(
                  offset + node.nodeSize,
                  (view) =>
                    buildWidget(code, i, () =>
                      view.dispatch(view.state.tr.setMeta(widgetsKey, { toggle: i }))
                    ),
                  // rebuilt only when the code changes (a rebuild restarts the widget), not on every keystroke elsewhere
                  {
                    side: -1,
                    key: `widget-${i}-${code.length}-${hash(code)}`,
                    ignoreSelection: true
                  }
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

function hash(s: string): number {
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) | 0;
  return h >>> 0;
}
