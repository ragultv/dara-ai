import { Extension } from "@tiptap/react";
import { Plugin, PluginKey } from "@tiptap/pm/state";
import { Decoration, DecorationSet, type EditorView } from "@tiptap/pm/view";

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

/**
 * What a widget looks like before its own CSS says anything: the page's colours (as variables, also updated live when the theme changes)
 * and a quiet default for buttons, fields, labels, tables and SVG, written with :where() so any style the widget writes itself wins.
 */
const BASE_CSS = [
  "html,body{margin:0}",
  'body{padding:4px 2px;font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--text);background:transparent}',
  ":where(h1,h2,h3,h4){margin:0 0 .5em;line-height:1.25;font-weight:650}",
  ":where(p){margin:.4em 0}",
  ":where(button,input,select,textarea){font:inherit;color:inherit}",
  ":where(button){cursor:pointer;border:1px solid transparent;border-radius:8px;padding:.5em 1em;background:var(--primary);color:var(--on-primary)}",
  ":where(button:hover){background:var(--primary-hover)}",
  ":where(button:disabled){opacity:.5;cursor:not-allowed}",
  ":where(input,select,textarea){border:1px solid var(--border);border-radius:8px;padding:.5em .7em;background:var(--bg);color:var(--text);box-sizing:border-box}",
  ":where(input:focus,select:focus,textarea:focus){outline:2px solid var(--primary);outline-offset:1px}",
  ":where(label){display:block;margin:.6em 0 .25em;font-weight:600;color:var(--muted);font-size:.9em}",
  ":where(input[type=range]){width:100%;accent-color:var(--primary)}",
  ":where(table){border-collapse:collapse;width:100%}",
  ":where(th,td){border-bottom:1px solid var(--border);padding:.5em .6em;text-align:left}",
  ":where(th){color:var(--muted);font-weight:600}",
  ":where(svg){max-width:100%;height:auto}",
  ":where(svg text){fill:var(--text);font:13px system-ui,-apple-system,sans-serif}",
  ":where(a){color:var(--primary)}"
].join("");

const THEME_KEYS = [
  "text",
  "muted",
  "bg",
  "surface",
  "border",
  "primary",
  "primary-hover",
  "on-primary"
];

function cssVars(colors: Record<string, string>): string {
  return THEME_KEYS.filter((k) => colors[k])
    .map((k) => `--${k}:${colors[k]!.replace(/[<>;{}]/g, "")}`)
    .join(";");
}

/** The page the frame loads. The Content-Security-Policy blocks every outside request; only inline script and style run. */
export function widgetSrcDoc(
  code: string,
  colors: Record<string, string>,
  frameId: string
): string {
  const csp =
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; media-src data:; font-src data:";
  // the frame reports its height, and takes new colours when the app's theme changes (only from the page that holds it)
  const glue =
    `(function(){var id=${JSON.stringify(frameId)};` +
    `var send=function(){parent.postMessage({daraWidgetHeight:document.documentElement.scrollHeight,frame:id},"*")};` +
    `new ResizeObserver(send).observe(document.documentElement);window.addEventListener("load",send);send();` +
    `window.addEventListener("message",function(e){if(e.source!==parent)return;var t=e.data&&e.data.daraTheme;if(!t||typeof t!=="object")return;` +
    `for(var k in t){if(typeof t[k]==="string"&&/^[a-z-]+$/.test(k)&&k!=="scheme")document.documentElement.style.setProperty("--"+k,t[k].replace(/[<>;{}]/g,""))}` +
    `if(t.scheme==="dark"||t.scheme==="light")document.documentElement.style.colorScheme=t.scheme;send()});})();`;
  return [
    '<!doctype html><html><head><meta charset="utf-8">',
    `<meta http-equiv="Content-Security-Policy" content="${csp}">`,
    `<style>:root{${cssVars(colors)};color-scheme:${colors["scheme"] === "dark" ? "dark" : "light"}}${BASE_CSS}</style>`,
    "</head><body>",
    code,
    `<script>${glue}</script>`,
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
  // a theme change recolours every widget on the page without restarting it
  new MutationObserver(() => {
    const colors = themeColors();
    for (const [id, frame] of frames) {
      if (!frame.isConnected) frames.delete(id);
      else frame.contentWindow?.postMessage({ daraTheme: colors }, "*");
    }
  }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  // any click outside a widget menu closes it
  document.addEventListener("mousedown", (e) => {
    document.querySelectorAll<HTMLElement>(".doc-widget-menu[data-open='true']").forEach((m) => {
      if (!m.parentElement?.contains(e.target as Node)) m.dataset.open = "false";
    });
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
    "primary-hover": get("primary-hover") || "#5b21b6",
    "on-primary": get("on-primary") || "#fff",
    scheme: document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light"
  };
}

const DOTS =
  '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true"><circle cx="5" cy="12" r="1.7"/><circle cx="12" cy="12" r="1.7"/><circle cx="19" cy="12" r="1.7"/></svg>';

let counter = 0;
function buildWidget(
  code: string,
  index: number,
  actions: { toggle: () => void; remove: () => void }
): HTMLElement {
  listen();
  const root = document.createElement("div");
  root.className = "doc-widget";
  root.contentEditable = "false";

  // like Claude's: no header bar, a small "..." that appears on hover and opens Show code / Copy code / Delete widget
  const more = document.createElement("button");
  more.type = "button";
  more.className = "doc-widget-more";
  more.setAttribute("aria-label", "Widget options");
  more.setAttribute("aria-haspopup", "menu");
  more.innerHTML = DOTS;
  const menu = document.createElement("div");
  menu.className = "doc-widget-menu";
  menu.setAttribute("role", "menu");
  menu.dataset.open = "false";
  const item = (label: string, onClick: () => void, extra = "") => {
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("role", "menuitem");
    b.className = `doc-widget-item ${extra}`.trim();
    b.innerHTML = label;
    b.addEventListener("mousedown", (e) => e.preventDefault());
    b.addEventListener("click", () => {
      menu.dataset.open = "false";
      onClick();
    });
    return b;
  };
  // the label follows the code block's state through CSS (.widget-code-open + .doc-widget), so the frame is not rebuilt when it flips
  menu.append(
    item(
      '<span class="wl-show">Show code</span><span class="wl-hide">Hide code</span>',
      actions.toggle
    ),
    item("Copy code", () => void navigator.clipboard?.writeText(code).catch(() => {})),
    item("Delete widget", actions.remove, "danger")
  );
  more.addEventListener("mousedown", (e) => e.preventDefault());
  more.addEventListener("click", () => {
    menu.dataset.open = menu.dataset.open === "true" ? "false" : "true";
  });

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
  root.append(more, menu, frame);
  return root;
}

export const Widgets = Extension.create({
  name: "widgets",
  addProseMirrorPlugins() {
    let viewRef: EditorView | null = null;
    return [
      new Plugin<WidgetState>({
        key: widgetsKey,
        view: (v) => {
          viewRef = v;
          return {
            destroy() {
              viewRef = null;
            }
          };
        },
        state: {
          init: () => ({ shown: new Set<number>() }),
          apply(tr, value) {
            const t = tr.getMeta(widgetsKey) as { toggle?: number } | undefined;
            if (t?.toggle === undefined) return value;
            const next = new Set(value.shown);
            if (next.has(t.toggle)) next.delete(t.toggle);
            else next.add(t.toggle);
            return { shown: next };
          }
        },
        props: {
          // focus moving in or out of a widget's code opens or closes it, so redraw then
          handleDOMEvents: {
            focus: (view) => {
              view.dispatch(view.state.tr.setMeta(widgetsKey, { refresh: true }));
              return false;
            },
            blur: (view) => {
              view.dispatch(view.state.tr.setMeta(widgetsKey, { refresh: true }));
              return false;
            }
          },
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
              const inside =
                (viewRef?.hasFocus() ?? false) &&
                sel.from > offset &&
                sel.to < offset + node.nodeSize;
              const open = shown.has(i) || inside;
              decos.push(
                Decoration.node(offset, offset + node.nodeSize, {
                  class: open ? "widget-code widget-code-open" : "widget-code"
                })
              );
              decos.push(
                Decoration.widget(
                  offset + node.nodeSize,
                  (view, getPos) =>
                    buildWidget(code, i, {
                      toggle: () => view.dispatch(view.state.tr.setMeta(widgetsKey, { toggle: i })),
                      remove: () => {
                        // the widget sits right after its code block: remove that block (and with it, the widget)
                        const pos = getPos();
                        if (pos === undefined) return;
                        const before = view.state.doc.resolve(pos).nodeBefore;
                        if (before) view.dispatch(view.state.tr.delete(pos - before.nodeSize, pos));
                      }
                    }),
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
