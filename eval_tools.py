"""Review eval questions in Excel instead of editing JSONL by hand.

Step 1 - draft questions (uses your LLM key):
    python -m src.gen_eval_candidates --n 90

Step 2 - make the review workbook (one-time: pip install openpyxl):
    python eval_tools.py to-xlsx
    -> eval\\review.xlsx   (open it in Excel; it is already formatted)

Step 3 - review in Excel. For each row:
    keep       pick y in the dropdown to keep the question; leave blank to drop it
    question   fix the wording; rewrite about half of them in your own words
    answer     fix it so it is correct and short
    evidence   the passage the draft came from (help for checking, read-only)
    doc/pages  the PDF and the page number(s) holding the answer: 4  or  4;5  or  4-5
    The yellow rows at the bottom (u01, u02, ...) are for UNANSWERABLE questions: write a plausible
    question the documents cannot answer and pick y in keep. Leave doc, pages and answer empty.
    Save normally (Ctrl+S, keep the .xlsx format) and close Excel.

Step 4 - build the final file used by the evaluation:
    python eval_tools.py from-xlsx
    -> eval\\questions.jsonl

A CSV version also exists: to-csv / from-csv.

Unanswerable questions without a spreadsheet:
    1. Check a topic is really absent from your documents:   python eval_tools.py mentions mudra
    2. Write one question per line in eval\\unanswerable.txt (Notepad, UTF-8)
    3. python eval_tools.py add-unanswerable
"""
import csv
import io
import json
import re
import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parent / "eval"
CANDIDATES = EVAL / "candidates.jsonl"
REVIEW = EVAL / "review.csv"
XLSX = EVAL / "review.xlsx"
FINAL = EVAL / "questions.jsonl"
FIELDS = ["id", "keep", "question", "answer", "doc", "pages", "answerable", "language", "evidence"]
XLSX_FIELDS = ["id", "keep", "question", "answer", "evidence", "doc", "pages", "answerable", "language"]
KEEP_VALUES = {"y", "yes", "1", "x", "true"}
N_UNANSWERABLE_ROWS = 15


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def load_candidates() -> list[dict]:
    if not CANDIDATES.exists():
        sys.exit(f"{CANDIDATES} not found. Run: python -m src.gen_eval_candidates --n 90")
    rows = []
    with CANDIDATES.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append({
                "id": r["id"], "keep": "", "question": r["question"], "answer": r["answer"],
                "doc": r["doc"], "pages": ";".join(str(p) for p in r["pages"]),
                "answerable": "TRUE", "language": r.get("language", "en"),
                "evidence": r.get("evidence", ""),
            })
    return rows


def parse_pages(text: str) -> list[int]:
    """'4' -> [4]; '4;5' or '4,5' -> [4, 5]; '4-6' -> [4, 5, 6]."""
    pages: list[int] = []
    for tok in re.split(r"[;,\s]+", text.strip()):
        m = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", tok)
        if m and 0 <= int(m.group(2)) - int(m.group(1)) <= 20:
            pages += list(range(int(m.group(1)), int(m.group(2)) + 1))
        elif tok.isdigit():
            pages.append(int(tok))
    return pages


def finish(rows: list[dict], columns: list[str], extra: str = "", how_to_save: str = "") -> None:
    """Turn reviewed rows into eval/questions.jsonl (shared by the CSV and Excel paths)."""
    kept, problems = [], []
    filled = sum(1 for r in rows if (r.get("keep") or "").strip())
    for n, row in enumerate(rows, start=2):  # row 1 is the header
        if (row.get("keep") or "").strip().lower() not in KEEP_VALUES:
            continue
        q = (row.get("question") or "").strip()
        if not q:
            problems.append(f"row {n}: empty question")
            continue
        answerable = (row.get("answerable") or "TRUE").strip().lower() not in {"false", "0", "no", "n"}
        item = {"id": (row.get("id") or f"r{n}").strip() or f"r{n}", "question": q,
                "answerable": answerable, "language": (row.get("language") or "en").strip(),
                "verified": True}
        if answerable:
            pages = parse_pages(row.get("pages") or "")
            doc = (row.get("doc") or "").strip()
            ans = (row.get("answer") or "").strip()
            if not (doc and pages and ans):
                problems.append(f"row {n} ({item['id']}): answerable question needs doc, pages and answer")
                continue
            item.update(answer=ans, doc=doc, pages=pages)
        kept.append(item)
    if problems:
        print("Fix these rows and run again:\n  " + "\n  ".join(problems))
        sys.exit(1)
    if not kept:
        print(f"No questions were kept, so {FINAL.name} was NOT changed.\n"
              f"  rows read: {len(rows)}; rows with something in 'keep': {filled}{extra}\n"
              f"  columns found: {columns}\n"
              f"  Mark the good rows with y in the 'keep' column, {how_to_save}run again.")
        sys.exit(1)
    with FINAL.open("w", encoding="utf-8") as f:
        for item in kept:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    n_un = sum(1 for k in kept if not k["answerable"])
    print(f"Wrote {len(kept)} questions ({len(kept) - n_un} answerable, {n_un} unanswerable) to {FINAL}")


# --------------------------------------------------------------------------- #
# Excel workbook (recommended)
# --------------------------------------------------------------------------- #
def _cell_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if hasattr(v, "isoformat"):  # a date Excel created from typed text
        return v.isoformat()[:10]
    return str(v)


def to_xlsx():
    try:
        import openpyxl
    except ImportError:
        sys.exit("Run this once first:  pip install openpyxl")
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    if XLSX.exists():
        sys.exit(f"{XLSX} already exists, so I did not overwrite your work.\n"
                 "To start over, rename or delete that file and run this again.")
    rows = load_candidates()
    n_drafts = len(rows)
    for i in range(1, N_UNANSWERABLE_ROWS + 1):
        rows.append({"id": f"u{i:02d}", "keep": "", "question": "", "answer": "", "doc": "",
                     "pages": "", "answerable": "FALSE", "language": "en", "evidence": ""})

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "review"
    ws.append(XLSX_FIELDS)
    for r in rows:
        ws.append([r[k] for k in XLSX_FIELDS])

    col = {name: i + 1 for i, name in enumerate(XLSX_FIELDS)}
    letter = {name: openpyxl.utils.get_column_letter(i) for name, i in col.items()}
    widths = {"id": 7, "keep": 7, "question": 48, "answer": 38, "evidence": 80,
              "doc": 24, "pages": 10, "answerable": 11, "language": 9}
    for name, w in widths.items():
        ws.column_dimensions[letter[name]].width = w

    last = len(rows) + 1
    top_wrap = Alignment(wrap_text=True, vertical="top")
    header_fill = PatternFill("solid", fgColor="1F3864")
    unans_fill = PatternFill("solid", fgColor="FFF2CC")
    text_cols = ["question", "answer", "evidence", "doc", "pages"]
    for r in range(1, last + 31):  # 30 spare rows below, formatted too
        for name in XLSX_FIELDS:
            c = ws.cell(row=r, column=col[name])
            c.alignment = top_wrap
            if r > 1 and name in text_cols:
                c.number_format = "@"  # text: stops Excel turning "15.08.2019" or "4-5" into dates
        if r == 1:
            ws.row_dimensions[r].height = 22
            for name in XLSX_FIELDS:
                c = ws.cell(row=r, column=col[name])
                c.font = Font(bold=True, color="FFFFFF")
                c.fill = header_fill
        elif r <= n_drafts + 1:
            ws.row_dimensions[r].height = 110  # about 7 lines; click a cell to read the rest
        elif r <= last:
            ws.row_dimensions[r].height = 36
            for name in XLSX_FIELDS:
                ws.cell(row=r, column=col[name]).fill = unans_fill

    keep_dv = DataValidation(type="list", formula1='"y"', allow_blank=True)
    ans_dv = DataValidation(type="list", formula1='"TRUE,FALSE"', allow_blank=True)
    ws.add_data_validation(keep_dv)
    ws.add_data_validation(ans_dv)
    keep_dv.add(f"{letter['keep']}2:{letter['keep']}{last + 30}")
    ans_dv.add(f"{letter['answerable']}2:{letter['answerable']}{last + 30}")
    ws.freeze_panes = "C2"  # keep id and keep visible while scrolling
    wb.save(XLSX)
    print(f"Wrote {n_drafts} draft questions + {N_UNANSWERABLE_ROWS} blank rows for unanswerable questions "
          f"to {XLSX}\nOpen it in Excel, review, save (Ctrl+S), close Excel, then run: "
          "python eval_tools.py from-xlsx")


def from_xlsx():
    try:
        import openpyxl
    except ImportError:
        sys.exit("Run this once first:  pip install openpyxl")
    if not XLSX.exists():
        sys.exit(f"{XLSX} not found. Run: python eval_tools.py to-xlsx")
    try:
        wb = openpyxl.load_workbook(XLSX, data_only=True)
    except PermissionError:
        sys.exit("Cannot read review.xlsx. Close it in Excel first, then run this again.")
    ws = wb["review"] if "review" in wb.sheetnames else wb.active
    it = ws.iter_rows(values_only=True)
    header = [_cell_text(h).strip() for h in next(it, [])]
    rows = [{h: _cell_text(v) for h, v in zip(header, vals)} for vals in it]
    rows = [r for r in rows if any(r.values())]
    finish(rows, header, how_to_save="save the workbook (Ctrl+S), close Excel, and ")


# --------------------------------------------------------------------------- #
# CSV version (kept as a fallback)
# --------------------------------------------------------------------------- #
def to_csv():
    rows = load_candidates()
    with REVIEW.open("w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig so Excel reads it right
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {REVIEW}. Open it in Excel, review, save as CSV UTF-8, then run from-csv.")


def from_csv():
    if not REVIEW.exists():
        sys.exit(f"{REVIEW} not found. Run: python eval_tools.py to-csv")
    try:
        text = REVIEW.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        sys.exit("review.csv is not UTF-8. In Excel use Save As -> 'CSV UTF-8 (Comma delimited)'.")
    header = text.splitlines()[0] if text else ""
    delim = max([",", ";", "\t"], key=header.count)  # some Excel setups save with ; or tabs
    columns = next(csv.reader([header], delimiter=delim), [])
    with io.StringIO(text, newline="") as f:
        rows = list(csv.DictReader(f, delimiter=delim))
    finish(rows, columns, extra=f"; separator: {delim!r}",
           how_to_save="save as CSV UTF-8, close Excel, and ")


# --------------------------------------------------------------------------- #
# Unanswerable questions without a spreadsheet
# --------------------------------------------------------------------------- #
UNANSWERABLE_TXT = EVAL / "unanswerable.txt"
CHUNKS = EVAL.parent / "data" / "processed" / "chunks_fixed.jsonl"


def add_unanswerable():
    """Read eval/unanswerable.txt (one question per line) and add them to questions.jsonl."""
    if not FINAL.exists():
        sys.exit(f"{FINAL} not found. Run from-csv (or from-xlsx) first.")
    if not UNANSWERABLE_TXT.exists():
        sys.exit(f"{UNANSWERABLE_TXT} not found. Create it in Notepad: one question per line.")
    try:
        lines = UNANSWERABLE_TXT.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError:
        sys.exit("unanswerable.txt is not UTF-8. In Notepad use Save As -> Encoding: UTF-8.")
    questions = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    if not questions:
        sys.exit("unanswerable.txt has no questions.")
    items = []
    with FINAL.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item.get("answerable", True):  # drop old unanswerable rows so re-running never duplicates
                    items.append(item)
    for i, q in enumerate(questions, start=1):
        items.append({"id": f"u{i:02d}", "question": q, "answerable": False, "language": "en", "verified": True})
    with FINAL.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    n_un = len(questions)
    print(f"{FINAL.name} now has {len(items)} questions ({len(items) - n_un} answerable, {n_un} unanswerable)")


def mentions():
    """Check whether a word or phrase appears anywhere in your documents:  python eval_tools.py mentions mudra"""
    if len(sys.argv) < 3:
        sys.exit("Usage: python eval_tools.py mentions <word or phrase>")
    if not CHUNKS.exists():
        sys.exit(f"{CHUNKS} not found. Run: python -m src.ingest")
    word = " ".join(sys.argv[2:]).lower()
    hits = []
    with CHUNKS.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if word in c["text"].lower():
                hits.append((c["doc"], c["page_start"]))
    print(f"'{word}' appears in {len(hits)} chunks")
    for doc, page in sorted(set(hits))[:10]:
        print(f"   {doc}  p.{page}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    commands = {"to-xlsx": to_xlsx, "from-xlsx": from_xlsx, "to-csv": to_csv, "from-csv": from_csv,
                "add-unanswerable": add_unanswerable, "mentions": mentions}
    commands.get(cmd, lambda: sys.exit(__doc__))()
