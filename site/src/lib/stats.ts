export interface Row {
  task: string;
  category: string;
  promptTokens: number | null;
  llama: number | null;
  plain: number | null;
  flash: number | null;
  ttftLlama: number | null;
  ttftDara: number | null;
  sameTextAsPlain: boolean | null;
}

/** Dara's decode speed divided by llama.cpp's, per task, for plain decoding or Flash. */
export function ratios(rows: Row[], kind: "plain" | "flash"): number[] {
  return rows.filter((r) => r.llama && r[kind]).map((r) => (r[kind] as number) / (r.llama as number));
}

export function geomean(xs: number[]): number | null {
  const v = xs.filter((x) => x > 0);
  return v.length ? Math.exp(v.reduce((a, x) => a + Math.log(x), 0) / v.length) : null;
}

export function categoryMean(rows: Row[], category: string, kind: "plain" | "flash"): number | null {
  return geomean(ratios(rows.filter((r) => r.category === category), kind));
}

export const CATEGORY_LABEL: Record<string, string> = {
  edit: "Edits",
  transform: "Rewrites and conversions",
  generate: "Writing from scratch"
};

export const TASK_LABEL: Record<string, string> = {
  function_edit: "Edit a function",
  json_add_field: "Add a field to JSON",
  sql_rename: "Rename a SQL column",
  fix_typos: "Fix typos",
  csv_to_table: "CSV to table",
  invoice_to_json: "Invoice to JSON",
  data_to_table: "Data to table",
  repeated_transcript: "Clean up a transcript",
  write_function: "Write a function",
  short_email: "Write a short email",
  tone_rewrite: "Rewrite in another tone",
  project_plan: "Project plan",
  notes_to_todos: "Meeting notes to to-dos",
  summary: "Summarize a text",
  tamil_summary: "Summarize (Tamil)",
  tamil_email: "Write an email (Tamil)"
};
