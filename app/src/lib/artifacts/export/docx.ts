import {
  AlignmentType,
  BorderStyle,
  Document,
  ExternalHyperlink,
  HeadingLevel,
  ImageRun,
  LevelFormat,
  Packer,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableRow,
  TextRun,
  WidthType,
  type ParagraphChild
} from "docx";

// Word export. The document's structure (headings, lists, tables, code, quotes, links, images) is turned into a real .docx, set in the
// same look as the page: a serif body, bold headings, a shaded header row on tables, shaded monospace code, thin rules.

export interface PMNode {
  type: string;
  attrs?: Record<string, unknown>;
  content?: PMNode[];
  text?: string;
  marks?: Array<{ type: string; attrs?: Record<string, unknown> }>;
}

export interface ExportImage {
  data: Uint8Array;
  type: "png" | "jpg" | "gif";
  width: number;
  height: number;
}

export interface DocxOptions {
  title: string;
  /** the stored bytes of an image (an asset id), or null when it cannot be shown in Word (WebP, a missing file) */
  loadImage: (assetId: string) => Promise<ExportImage | null>;
}

const BODY_FONT = "Palatino Linotype";
const CODE_FONT = "Consolas";
const INK = "1A1A1A";
const MUTED = "5F6368";
const RULE = "D5D7DB";
const HEADER_FILL = "F1F2F4";
const CODE_FILL = "F4F5F6";
const LINK = "1D4ED8";
const MAX_IMAGE_WIDTH = 560; // pixels, about 5.8 inches
const ASSET_RE = /^dara-asset:([0-9a-f-]{36})$/i;

interface RunStyle {
  bold?: boolean;
  italics?: boolean;
  size?: number;
  color?: string;
  font?: string;
}

const text = (t: string, s: RunStyle = {}) =>
  new TextRun({
    text: t,
    font: s.font ?? BODY_FONT,
    size: s.size ?? 22,
    color: s.color ?? INK,
    ...(s.bold ? { bold: true } : {}),
    ...(s.italics ? { italics: true } : {})
  });

/** The text of a node with its marks as Word formatting. */
function inline(nodes: PMNode[] | undefined, base: RunStyle = {}): ParagraphChild[] {
  const out: ParagraphChild[] = [];
  for (const n of nodes ?? []) {
    if (n.type === "hardBreak") {
      out.push(new TextRun({ break: 1 }));
      continue;
    }
    if (n.type !== "text" || !n.text) continue;
    const marks = new Set((n.marks ?? []).map((m) => m.type));
    const style: RunStyle = { ...base };
    if (marks.has("bold")) style.bold = true;
    if (marks.has("italic")) style.italics = true;
    const link = n.marks?.find((m) => m.type === "link")?.attrs?.["href"];
    const code = marks.has("code");
    const run = new TextRun({
      text: n.text,
      font: code ? CODE_FONT : (style.font ?? BODY_FONT),
      size: code ? 20 : (style.size ?? 22),
      color: typeof link === "string" ? LINK : (style.color ?? INK),
      ...(style.bold ? { bold: true } : {}),
      ...(style.italics ? { italics: true } : {}),
      ...(marks.has("strike") ? { strike: true } : {}),
      ...(typeof link === "string" ? { underline: {} } : {}),
      ...(code ? { shading: { type: ShadingType.CLEAR, fill: CODE_FILL, color: "auto" } } : {})
    });
    out.push(
      typeof link === "string" && /^(https?:|mailto:)/i.test(link)
        ? new ExternalHyperlink({ link, children: [run] })
        : run
    );
  }
  return out;
}

const plain = (nodes: PMNode[] | undefined): string =>
  (nodes ?? []).map((n) => (n.type === "text" ? (n.text ?? "") : plain(n.content))).join("");

const HEADINGS = [
  { level: HeadingLevel.HEADING_1, size: 38, before: 360, after: 160 },
  { level: HeadingLevel.HEADING_2, size: 30, before: 320, after: 120 },
  { level: HeadingLevel.HEADING_3, size: 25, before: 260, after: 100 }
];

interface Ctx {
  opts: DocxOptions;
  /** a counter so every numbered list starts again at 1 */
  lists: { n: number };
}

type Block = Paragraph | Table;

async function blocks(
  nodes: PMNode[] | undefined,
  ctx: Ctx,
  level = 0,
  quote = false
): Promise<Block[]> {
  const out: Block[] = [];
  for (const n of nodes ?? []) out.push(...(await block(n, ctx, level, quote)));
  return out;
}

const quoteLook = (quote: boolean) =>
  quote
    ? {
        indent: { left: 360 },
        border: { left: { style: BorderStyle.SINGLE, size: 12, space: 8, color: RULE } }
      }
    : {};

async function block(n: PMNode, ctx: Ctx, level: number, quote: boolean): Promise<Block[]> {
  switch (n.type) {
    case "paragraph": {
      const kids = inline(n.content, quote ? { color: MUTED, italics: true } : {});
      return [
        new Paragraph({
          children: kids,
          spacing: { after: 160, line: 312 },
          ...quoteLook(quote)
        })
      ];
    }
    case "heading": {
      const h = HEADINGS[Math.min(2, Math.max(0, Number(n.attrs?.["level"] ?? 1) - 1))]!;
      return [
        new Paragraph({
          heading: h.level,
          keepNext: true,
          spacing: { before: h.before, after: h.after },
          children: inline(n.content, { bold: true, size: h.size })
        })
      ];
    }
    case "bulletList":
    case "orderedList":
    case "taskList": {
      const reference = n.type === "orderedList" ? "numbers" : "bullets";
      const instance = n.type === "orderedList" ? ++ctx.lists.n : 0;
      const out: Block[] = [];
      for (const item of n.content ?? []) {
        const checked = item.attrs?.["checked"] === true;
        let first = true;
        for (const child of item.content ?? []) {
          if (child.type === "paragraph") {
            const prefix = n.type === "taskList" && first ? [text(checked ? "☑ " : "☐ ")] : [];
            out.push(
              new Paragraph({
                children: [...prefix, ...inline(child.content)],
                spacing: { after: 80, line: 300 },
                ...(first
                  ? { numbering: { reference, level: Math.min(level, 5), instance } }
                  : { indent: { left: 720 * (level + 1) } })
              })
            );
          } else out.push(...(await block(child, ctx, level + 1, quote)));
          first = false;
        }
      }
      return out;
    }
    case "blockquote":
      return blocks(n.content, ctx, level, true);
    case "codeBlock": {
      if (n.attrs?.["language"] === "widget")
        return [
          new Paragraph({
            spacing: { after: 160 },
            children: [
              text("[Interactive widget: open this document in Dara to use it.]", {
                italics: true,
                color: MUTED
              })
            ]
          })
        ];
      const lines = plain(n.content).split("\n");
      return lines.map(
        (line, i) =>
          new Paragraph({
            shading: { type: ShadingType.CLEAR, fill: CODE_FILL, color: "auto" },
            spacing: {
              before: i === 0 ? 120 : 0,
              after: i === lines.length - 1 ? 200 : 0,
              line: 260
            },
            indent: { left: 160, right: 160 },
            children: [new TextRun({ text: line || " ", font: CODE_FONT, size: 18, color: INK })]
          })
      );
    }
    case "horizontalRule":
      return [
        new Paragraph({
          spacing: { before: 160, after: 160 },
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, space: 1, color: RULE } }
        })
      ];
    case "table":
      return [await table(n, ctx)];
    case "image": {
      const src = String(n.attrs?.["src"] ?? "");
      const alt = String(n.attrs?.["alt"] ?? "");
      const id = ASSET_RE.exec(src)?.[1];
      if (n.attrs?.["title"] === "video" || !id)
        return [
          new Paragraph({
            spacing: { after: 160 },
            children: [
              text(
                `[${n.attrs?.["title"] === "video" ? "Video" : "Image"}: ${alt || "not included"}]`,
                { italics: true, color: MUTED }
              )
            ]
          })
        ];
      const img = await ctx.opts.loadImage(id);
      if (!img)
        return [
          new Paragraph({
            spacing: { after: 160 },
            children: [
              text(`[Image: ${alt || "could not be included"}]`, { italics: true, color: MUTED })
            ]
          })
        ];
      const scale = Math.min(1, MAX_IMAGE_WIDTH / img.width);
      return [
        new Paragraph({
          spacing: { before: 120, after: 200 },
          children: [
            new ImageRun({
              type: img.type,
              data: img.data,
              transformation: {
                width: Math.round(img.width * scale),
                height: Math.round(img.height * scale)
              },
              altText: { name: alt || "image", description: alt || "image", title: alt || "image" }
            })
          ]
        })
      ];
    }
    default:
      return blocks(n.content, ctx, level, quote);
  }
}

const cellBorder = { style: BorderStyle.SINGLE, size: 4, color: RULE };

async function table(n: PMNode, ctx: Ctx): Promise<Table> {
  const rows: TableRow[] = [];
  let widths: number[] | null = null;
  for (const [ri, row] of (n.content ?? []).entries()) {
    const cells: TableCell[] = [];
    for (const cell of row.content ?? []) {
      const header = cell.type === "tableHeader";
      const inner = await blocks(cell.content, ctx);
      cells.push(
        new TableCell({
          children: inner.length ? (inner as Paragraph[]) : [new Paragraph({})],
          margins: { top: 90, bottom: 90, left: 140, right: 140 },
          borders: { top: cellBorder, bottom: cellBorder, left: cellBorder, right: cellBorder },
          ...(header
            ? { shading: { type: ShadingType.CLEAR, fill: HEADER_FILL, color: "auto" } }
            : {})
        })
      );
    }
    if (ri === 0) {
      const w = (row.content ?? []).map((c) => {
        const cw = c.attrs?.["colwidth"];
        return Array.isArray(cw) && typeof cw[0] === "number" ? cw[0] * 15 : 0; // pixels to twips
      });
      if (w.length && w.every((x) => x > 0)) widths = w;
    }
    rows.push(
      new TableRow({ children: cells, tableHeader: row.content?.[0]?.type === "tableHeader" })
    );
  }
  return new Table({
    rows,
    width: { size: 100, type: WidthType.PERCENTAGE },
    ...(widths ? { columnWidths: widths } : {})
  });
}

const levels = (
  format: (typeof LevelFormat)[keyof typeof LevelFormat],
  text: (l: number) => string
) =>
  Array.from({ length: 6 }, (_, level) => ({
    level,
    format,
    text: text(level),
    alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 720 * (level + 1), hanging: 360 } } }
  }));

/** The document as a .docx file: its title, then the body from the editor's JSON. */
export async function buildDocx(json: PMNode, opts: DocxOptions): Promise<Blob> {
  const ctx: Ctx = { opts, lists: { n: 0 } };
  const body = await blocks(json.content, ctx);
  const children: Block[] = [];
  if (opts.title.trim())
    children.push(
      new Paragraph({
        heading: HeadingLevel.TITLE,
        spacing: { after: 280 },
        children: [text(opts.title.trim(), { bold: true, size: 52 })]
      })
    );
  children.push(...body);
  const doc = new Document({
    title: opts.title || "Untitled",
    creator: "Dara",
    styles: { default: { document: { run: { font: BODY_FONT, size: 22, color: INK } } } },
    numbering: {
      config: [
        { reference: "bullets", levels: levels(LevelFormat.BULLET, () => "•") },
        { reference: "numbers", levels: levels(LevelFormat.DECIMAL, (l) => `%${l + 1}.`) }
      ]
    },
    sections: [
      {
        properties: { page: { margin: { top: 1300, bottom: 1300, left: 1440, right: 1440 } } },
        children
      }
    ]
  });
  return Packer.toBlob(doc);
}

/** Bytes and size of a stored image for Word (PNG, JPEG, GIF only), from the blob the server returned. */
export async function readExportImage(blob: Blob): Promise<ExportImage | null> {
  const type =
    blob.type === "image/png"
      ? "png"
      : blob.type === "image/jpeg"
        ? "jpg"
        : blob.type === "image/gif"
          ? "gif"
          : null;
  if (!type) return null;
  try {
    const bmp = await createImageBitmap(blob);
    const size = { width: bmp.width, height: bmp.height };
    bmp.close();
    return { data: new Uint8Array(await blob.arrayBuffer()), type, ...size };
  } catch {
    return null;
  }
}
