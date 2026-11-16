"""Self-instruct prompt generation for Dara Flash training data: the target model itself writes diverse, realistic user requests per category
(no external datasets, no downloads). Output: JSONL {"id", "prompt", "category"} -- the input of tools/flash_data_qwen / flash_data_gemma.

Usage: python -I tools/flash/self_instruct.py --gguf models/qwen3.5-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf --out build/flash/prompts_si.jsonl
       [--per 25] [--flash build/flash/drafter_4b.dflash] [--start-id 10000]
"""
import argparse, json, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXE = os.path.join(ROOT, "build", "cuda", "qwen_generate.exe")

CATEGORIES = [
    # writing
    "short professional emails (include the situation in the request)", "polite replies to customer complaints (include the complaint text)",
    "cover letters and job applications", "LinkedIn and social media posts", "product descriptions for an online shop", "blog post introductions",
    "short stories and creative writing with specific constraints", "poems on everyday topics", "speeches and toasts", "meeting agendas and minutes",
    "formal letters to government offices or banks", "apology messages", "invitations for events", "marketing slogans and ad copy",
    "newsletter sections", "personal statements for college admission", "reviews of books, films or restaurants",
    # editing / transforming text given in the request
    "fix grammar and spelling in a paragraph (include the paragraph)", "rewrite a paragraph in a more formal tone (include it)",
    "rewrite a paragraph to be shorter and clearer (include it)", "summarize a long paragraph in two sentences (include the paragraph)",
    "turn bullet notes into a well-written paragraph (include the notes)", "convert meeting notes into a to-do list (include the notes)",
    "extract names, dates and amounts from a text as JSON (include the text)", "convert CSV data into a markdown table (include the CSV)",
    "convert a list of facts into a JSON object (include the facts)", "reformat an address list into a consistent format (include it)",
    "change the tense of a paragraph (include it)", "make an email more friendly (include the email)", "add headings to a document (include it)",
    "correct the punctuation of a text message thread (include it)", "turn a recipe paragraph into numbered steps (include it)",
    # code
    "Python functions for everyday data tasks", "Python scripts for files and folders", "pandas data analysis questions", "SQL queries for business reports",
    "JavaScript functions for web pages", "TypeScript types and interfaces", "React components", "C++ functions and classes", "Java classes",
    "Go functions", "Rust functions", "shell and PowerShell one-liners", "regular expressions with explanations", "debugging a short code snippet (include the code)",
    "refactoring a short code snippet (include the code)", "writing unit tests for a given function (include the function)",
    "explaining what a code snippet does (include the code)", "REST API design questions", "git commands for common situations", "Docker and deployment questions",
    "algorithms and data structures explained", "time complexity questions", "HTML and CSS layout questions",
    # Tamil
    "requests written in Tamil asking for explanations of everyday topics", "requests written in Tamil asking for letters or applications",
    "requests written in Tamil asking for tips and advice", "requests written in Tamil about Tamil literature and culture",
    "translate an English paragraph into Tamil (include the paragraph)", "translate a Tamil paragraph into English (include the Tamil paragraph)",
    "translate short business messages into Tamil (include them)", "requests to summarize a Tamil paragraph (include it, written in Tamil)",
    # other languages / translation
    "translate English sentences into Hindi (include them)", "translate English sentences into French or Spanish (include them)",
    # explanations / knowledge
    "science questions for school students", "physics concepts explained simply", "chemistry questions", "biology and human body questions",
    "history questions about India", "world history questions", "geography questions", "economics and personal finance questions",
    "health and fitness advice questions", "nutrition and diet planning", "mental health and stress management", "parenting questions",
    "travel planning for Indian cities", "cooking recipes and techniques", "gardening and plants", "home repair and maintenance",
    "career advice", "interview preparation questions", "study tips and exam preparation", "math word problems to solve step by step",
    "algebra and geometry problems", "statistics and probability questions", "logic puzzles and riddles", "explaining technology to older people",
    "smartphone and laptop troubleshooting", "internet safety and privacy", "machine learning concepts", "cloud computing concepts",
    "electric vehicles and energy", "climate and environment", "law and consumer rights questions (general information)", "small business planning",
    "marketing strategy questions", "startup and fundraising questions", "project management questions", "customer support scripts",
    "HR policy drafting", "sales pitch writing", "event planning", "wedding and festival planning in India", "agriculture and farming questions",
    "sports training questions", "music and art questions", "movies and books recommendations with reasons", "philosophy questions",
    "comparisons between two products or options", "pros and cons lists", "checklists for everyday tasks", "step-by-step how-to guides",
    "brainstorming ideas for a given goal", "role-play as a teacher explaining a topic", "role-play as an interviewer asking questions",
    "data interpretation from a small table (include the table)", "spreadsheet formula questions (Excel / Google Sheets)",
]

ASK = ("Write {n} different, realistic requests that a user might send to an AI assistant, in the category: {c}.\n"
       "Make them specific and varied in length and style (some short, some detailed). If the category says to include text, code or data, put that "
       "content inside the request itself. Output ONLY the requests, one per line, numbered 1. to {n}. Do not add any other text.")


def chat(user):
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def generate(gguf, text, n_new, flash):
    pf = os.path.join(ROOT, "build", "flash", "_si_prompt.txt")
    with open(pf, "w", encoding="utf-8") as f: f.write(text)
    cmd = [EXE, "--gguf", gguf, "--prompt-file", pf, "--special", "--stop-at-eos", "--new", str(n_new), "--bench", "--kv", "4096",
           "--out", os.path.join(ROOT, "build", "flash", "_si")]
    if flash: cmd += ["--flash", flash, "--spec", "4", "--flash-min-p", "0.6"]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL, timeout=1800)
    out = p.stdout
    if "--- generated ---" not in out: return ""
    g = out.split("--- generated ---", 1)[1]
    g = re.sub(r"flash drafter[^\n]*? [\d.]+ s\n", "", g)   # status lines printed while the text streams
    for stop in ["<|im_end|>", "\n--- end", "flash n-gram:", "flash time:", "\ndecode wall", "\nwarm decode"]:
        g = g.split(stop, 1)[0]
    return g.strip("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per", type=int, default=25)
    ap.add_argument("--rounds", type=int, default=1, help="ask each category this many times (more variety)")
    ap.add_argument("--flash", default=None)
    ap.add_argument("--start-id", type=int, default=10000)
    a = ap.parse_args()
    seen, n_id = set(), a.start_id
    if os.path.exists(a.out):   # resume: keep what is there
        for l in open(a.out, encoding="utf-8"):
            if l.strip():
                j = json.loads(l); seen.add(j["prompt"].lower()); n_id = max(n_id, j["id"] + 1)
    done_cats = {}
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            if l.strip(): c = json.loads(l)["category"]; done_cats[c] = done_cats.get(c, 0) + 1
    with open(a.out, "a", encoding="utf-8") as fo:
        for r in range(a.rounds):
            for c in CATEGORIES:
                if done_cats.get(c, 0) >= a.per * (r + 1) * 0.6: continue
                g = generate(a.gguf, chat(ASK.format(n=a.per, c=c)), 40 * a.per, a.flash)
                got = 0
                items = []
                for line in g.split("\n"):   # numbered items; unnumbered lines continue the previous item (included data, code, tables)
                    m = re.match(r"^\s*\d+[.)]\s*(.+)$", line)
                    if m: items.append(m.group(1))
                    elif items and line.strip(): items[-1] += "\n" + line.strip()
                for q in items:
                    q = q.strip().strip('"').strip()
                    if len(q) < 15 or len(q) > 1500 or q.lower() in seen: continue
                    seen.add(q.lower())
                    fo.write(json.dumps({"id": n_id, "prompt": q, "category": c}, ensure_ascii=False) + "\n"); n_id += 1; got += 1
                fo.flush()
                print(f"[{r}] {got:3d} new  {c}", flush=True)
    print(f"total prompts: {len(seen)}")


if __name__ == "__main__":
    main()
