"""Build JV-Data.md from PDF with proper Markdown tables (pages 10+)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz

PDF_PATH = Path(r"c:\Users\syugo\Downloads\JV-Data仕様書_4.9.0.1.pdf")
OUT_PATH = Path(__file__).resolve().parents[1] / "docs" / "JV-Data.md"

PAGE_SECTIONS: dict[int, str] = {
    9: "## 2. データフォーマット（レコードフォーマット）",
    29: "## 3. 特記事項",
    38: "## 4. コード表",
    50: "## 5. データ種別一覧",
    52: "### JV-Link JVOpenメソッド option と dataspec の関係",
    53: "## 6. データ提供タイミング・提供単位",
}

TIMING_PAGE_RANGE = (53, 55)  # inclusive 0-based PDF page indices

TIMING_COLS: list[tuple[int, int, str]] = [
    (0, 155, "データ名称"),
    (155, 182, "種別ID"),
    (182, 212, "曜日"),
    (212, 245, "時間"),
    (245, 382, "提供及び更新タイミング"),
    (382, 492, "提供単位"),
    (492, 10_000, "提供期間"),
]

TIMING_ID_RE = re.compile(
    r"^(TOKU|RACE|DIFF|DIFN|BLOD|BLDN|MING|SNAP|SNPN|SLOP|WOOD|YSCH|HOSE|HOYU|COMM|TCOV|TCVN|RCOV|RCVN|0B\d{2})$"
)

CODE_TITLE_RE = re.compile(r"^\d{4}\.")

RECORD_ID_MAP: dict[str, str] = {
    "特別登録馬": "TK",
    "レース詳細": "RA",
    "馬毎レース情報": "SE",
    "払戻": "HR",
    "票数１": "H1",
    "票数1": "H1",
    "票数６": "H6",
    "票数6": "H6",
    "オッズ１": "O1",
    "オッズ1": "O1",
    "オッズ２": "O2",
    "オッズ2": "O2",
    "オッズ３": "O3",
    "オッズ3": "O3",
    "オッズ４": "O4",
    "オッズ4": "O4",
    "オッズ５": "O5",
    "オッズ5": "O5",
    "オッズ６": "O6",
    "オッズ6": "O6",
    "競走馬マスタ": "UM",
    "騎手マスタ": "KS",
    "調教師マスタ": "CH",
    "生産者マスタ": "BR",
    "馬主マスタ": "BN",
    "繁殖馬マスタ": "HN",
    "産駒マスタ": "SK",
    "出走別着度数": "CK",
    "レコードマスタ": "RC",
    "坂路調教": "HC",
    "競走馬市場取引価格": "HS",
    "馬名の意味由来": "HY",
    "開催スケジュール": "YS",
    "系統情報": "BT",
    "コース情報": "CS",
    "タイム型データマイニング予想": "DM",
    "対戦型データマイニング予想": "TM",
    "重勝式": "WF",
    "競走馬除外情報": "JG",
    "ウッドチップ調教": "WC",
    "馬体重": "WH",
    "天候馬場状態": "WE",
    "出走取消": "AV",
    "騎手変更": "JC",
    "発走時刻変更": "TC",
    "コース変更": "CC",
}


@dataclass
class BuildState:
    format_header: list[str] = field(default_factory=list)
    format_rows: list[list[str]] = field(default_factory=list)
    format_preamble: list[str] = field(default_factory=list)

    code_title: str = ""
    code_header: list[str] = field(default_factory=list)
    code_rows: list[list[str]] = field(default_factory=list)

    notes_rows: list[list[str]] = field(default_factory=list)


def esc_cell(value: str) -> str:
    if not value:
        return ""
    value = value.replace("|", "\\|")
    value = value.replace("\n", "<br>")
    value = re.sub(r"\s+", " ", value).strip()
    return value


_ZEN = str.maketrans("０１２３４５６７８９", "0123456789")


def parse_record_meta(title_cell: str) -> tuple[str, str | None]:
    title_cell = title_cell.strip()
    m_len = re.search(r"レコード長\s*(\d[\d,]*)\s*バイト", title_cell)
    rec_len = m_len.group(1) if m_len else None

    name_part = re.sub(r"レコード長.*", "", title_cell).strip()
    name_part = re.sub(r"\s+", "", name_part)

    num = ""
    m_num = re.match(r"^([０-９\d]+)[．.]", name_part)
    if m_num:
        num = m_num.group(1).translate(_ZEN) + ". "

    name = re.sub(r"^[０-９\d]+[．.]", "", name_part)

    rec_id = ""
    for key, rid in RECORD_ID_MAP.items():
        if key in name:
            rec_id = rid
            break

    heading = f"### {num}{name}"
    if rec_id:
        heading += f"（{rec_id}）"
    return heading, rec_len


def drop_empty_columns(header: list[str], data_rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    if not header:
        return header, data_rows
    keep = []
    for col in range(len(header)):
        if header[col]:
            keep.append(col)
            continue
        if any(col < len(row) and row[col] for row in data_rows):
            keep.append(col)
    if len(keep) == len(header):
        return header, data_rows
    new_header = [header[i] for i in keep]
    new_rows = []
    for row in data_rows:
        cells = row + [""] * (len(header) - len(row))
        new_rows.append([cells[i] for i in keep])
    return new_header, new_rows


def label_sub_columns(header: list[str], data_rows: list[list[str]]) -> list[str]:
    for i, h in enumerate(header):
        if h:
            continue
        vals = {row[i] for row in data_rows if i < len(row) and row[i]}
        if vals and all(len(v) <= 3 and v.isascii() for v in vals):
            header[i] = "小項"
    return header


def find_header_row(rows: list[list[str | None]]) -> int:
    for i, row in enumerate(rows):
        cells = [str(c or "").strip() for c in row]
        if cells and cells[0] == "項番":
            return i
        joined = "".join(cells)
        if "項番" in joined and "項目名" in joined:
            return i
    return -1


def render_table(header: list[str], data_rows: list[list[str]]) -> str:
    if not header:
        return ""
    align = ["---"] * len(header)
    if header[0] == "項番":
        align[0] = ":---:"
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(align) + " |",
    ]
    for cells in data_rows:
        if not any(cells):
            continue
        if cells[0].startswith("項番2.") or cells[0] in ('"○"', '"-"', '"△"'):
            continue
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def normalize_row(row: list[str | None], width: int) -> list[str]:
    cells = [esc_cell(str(c or "")) for c in row]
    if len(cells) < width:
        cells.extend([""] * (width - len(cells)))
    elif len(cells) > width:
        cells = cells[:width]
    return cells


def looks_like_format_row(row: list[str | None]) -> bool:
    c0 = str(row[0] or "").strip() if row else ""
    c1 = str(row[1] or "").strip() if len(row) > 1 else ""
    c3 = str(row[3] or "").strip() if len(row) > 3 else ""
    if c0.isdigit():
        return True
    if c3.startswith("<") and ">" in c3:
        return True
    if not c0 and c1 and len(c1) <= 3 and c1.replace(" ", "").isalnum():
        return True
    return False


def is_record_format_start(rows: list[list[str | None]]) -> bool:
    if not rows:
        return False
    title = str(rows[0][0] or "")
    return "レコード長" in title and find_header_row(rows) >= 0


def is_format_continuation(rows: list[list[str | None]]) -> bool:
    if not rows or is_record_format_start(rows):
        return False
    if find_header_row(rows) >= 0:
        return False
    return any(looks_like_format_row(r) for r in rows)


def is_notes_table(rows: list[list[str | None]]) -> bool:
    if not rows:
        return False
    hdr = [str(c or "").strip() for c in rows[0]]
    return hdr[:3] == ["項番", "項目名", "特記事項"] or (
        len(hdr) >= 3 and hdr[0] == "項番" and hdr[1] == "項目名" and "特記事項" in hdr[2]
    )


def is_notes_orphan(rows: list[list[str | None]]) -> bool:
    if len(rows) != 1:
        return False
    c0 = str(rows[0][0] or "").strip()
    c1 = str(rows[0][1] or "").strip() if len(rows[0]) > 1 else ""
    return bool(c0 and c0[0].isdigit() and c1)


def is_code_table_start(rows: list[list[str | None]]) -> bool:
    if not rows:
        return False
    title = str(rows[0][0] or "").strip()
    return bool(CODE_TITLE_RE.match(title))


def is_code_continuation(rows: list[list[str | None]]) -> bool:
    if not rows or is_code_table_start(rows):
        return False
    if find_header_row(rows) >= 0:
        return False
    for row in rows[:3]:
        c1 = str(row[1] or "").strip() if len(row) > 1 else ""
        if c1 and (c1.isdigit() or re.match(r"^[A-Z]\d$", c1) or re.match(r"^[A-Z]{2}$", c1)):
            return True
    return False


def build_code_header(rows: list[list[str | None]]) -> list[str]:
    r1 = [str(c or "").strip() for c in rows[1]]
    r2 = [str(c or "").strip() for c in rows[2]] if len(rows) > 2 else []
    width = max(len(r1), len(r2))
    headers: list[str] = []
    for i in range(width):
        h1 = r1[i] if i < len(r1) else ""
        h2 = r2[i] if i < len(r2) else ""
        if h1 == "バイト数":
            headers.append("バイト数")
        elif h1 == "値":
            headers.append("値")
        elif h1 == "内容" and h2:
            headers.append(h2)
        elif h2:
            headers.append(h2)
        else:
            headers.append(h1)
    while headers and not headers[-1]:
        headers.pop()
    return [esc_cell(h) for h in headers]


def extract_code_data_rows(rows: list[list[str | None]], start: int, width: int) -> list[list[str]]:
    out: list[list[str]] = []
    for row in rows[start:]:
        cells = normalize_row(row, width)
        if not any(cells):
            continue
        out.append(cells)
    return out


def parse_format_table(rows: list[list[str | None]]) -> tuple[list[str], list[str], list[list[str]]]:
    preamble: list[str] = []
    title_row = str(rows[0][0] or "").strip() if rows[0] else ""
    if "レコード長" in title_row:
        heading, rec_len = parse_record_meta(title_row)
        preamble.append(heading)
        preamble.append("")
        if rec_len:
            preamble.append(f"- **レコード長**: `{rec_len}` バイト")
            preamble.append("")

    header_idx = find_header_row(rows)
    header = [esc_cell(str(c or "")) for c in rows[header_idx]]
    while header and not header[-1]:
        header.pop()

    data_rows_raw = rows[header_idx + 1 :]
    data_cells = [normalize_row(r, len(header)) for r in data_rows_raw]
    header, data_cells = drop_empty_columns(header, data_cells)
    header = label_sub_columns(header, data_cells)
    return preamble, header, data_cells


def flush_format(state: BuildState, chunks: list[str]) -> None:
    if not state.format_header:
        return
    if state.format_preamble:
        chunks.extend(state.format_preamble)
    table = render_table(state.format_header, state.format_rows)
    if table:
        chunks.append(table)
        chunks.append("")
    state.format_header = []
    state.format_rows = []
    state.format_preamble = []


def flush_code(state: BuildState, chunks: list[str]) -> None:
    if not state.code_header:
        return
    if state.code_title:
        chunks.append(f"#### {state.code_title}")
        chunks.append("")
    table = render_table(state.code_header, state.code_rows)
    if table:
        chunks.append(table)
        chunks.append("")
    state.code_title = ""
    state.code_header = []
    state.code_rows = []


def flush_notes(state: BuildState, chunks: list[str]) -> None:
    if not state.notes_rows:
        return
    header = ["項番", "項目名", "特記事項"]
    table = render_table(header, state.notes_rows)
    if table:
        chunks.append(table)
        chunks.append("")
    state.notes_rows = []


def ingest_format_start(state: BuildState, rows: list[list[str | None]], chunks: list[str]) -> None:
    flush_format(state, chunks)
    preamble, header, data = parse_format_table(rows)
    state.format_preamble = preamble
    state.format_header = header
    state.format_rows = data


def ingest_format_continuation(state: BuildState, rows: list[list[str | None]]) -> None:
    if not state.format_header:
        return
    width = len(state.format_header)
    for row in rows:
        if not any(str(c or "").strip() for c in row):
            continue
        cells = normalize_row(row, width)
        if not any(cells):
            continue
        state.format_rows.append(cells)


def ingest_code_start(state: BuildState, rows: list[list[str | None]], chunks: list[str]) -> None:
    flush_code(state, chunks)
    state.code_title = str(rows[0][0] or "").strip()
    state.code_header = build_code_header(rows)
    state.code_rows = extract_code_data_rows(rows, 3, len(state.code_header))


def ingest_code_continuation(state: BuildState, rows: list[list[str | None]]) -> None:
    if not state.code_header:
        return
    width = len(state.code_header)
    state.code_rows.extend(extract_code_data_rows(rows, 0, width))


def ingest_notes_table(state: BuildState, rows: list[list[str | None]], chunks: list[str]) -> None:
    flush_notes(state, chunks)
    width = 3
    for row in rows[1:]:
        cells = normalize_row(row, width)
        if any(cells):
            state.notes_rows.append(cells)


def ingest_notes_orphan(state: BuildState, rows: list[list[str | None]]) -> None:
    cells = normalize_row(rows[0], 3)
    if any(cells):
        state.notes_rows.append(cells)


def generic_table_to_markdown(rows: list[list[str | None]]) -> str:
    if not rows:
        return ""
    header = [esc_cell(str(c or "")) for c in rows[0]]
    while header and not header[-1]:
        header.pop()
    if not header:
        return ""
    data_cells = [normalize_row(r, len(header)) for r in rows[1:]]
    header, data_cells = drop_empty_columns(header, data_cells)
    return render_table(header, data_cells)


def timing_col_index(x: float) -> int:
    for i, (lo, hi, _) in enumerate(TIMING_COLS):
        if lo <= x < hi:
            return i
    return len(TIMING_COLS) - 1


def page_lines(page: fitz.Page) -> list[tuple[float, float, str]]:
    lines: list[tuple[float, float, str]] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            text = "".join(span["text"] for span in line["spans"]).strip()
            if not text or re.fullmatch(r"\d{1,2}", text):
                continue
            bbox = line["bbox"]
            lines.append((bbox[1], bbox[0], text))
    return lines


def group_lines_by_row(lines: list[tuple[float, float, str]], y_tol: float = 5.0) -> list[list[tuple[float, float, str]]]:
    if not lines:
        return []
    ordered = sorted(lines, key=lambda t: (t[0], t[1]))
    groups: list[list[tuple[float, float, str]]] = [[ordered[0]]]
    ref_y = ordered[0][0]
    for item in ordered[1:]:
        y = item[0]
        if y - ref_y <= y_tol:
            groups[-1].append(item)
        else:
            groups.append([item])
            ref_y = y
    return groups


def row_group_to_cells(group: list[tuple[float, float, str]]) -> list[str]:
    buckets: list[list[str]] = [[] for _ in TIMING_COLS]
    for _y, x, text in sorted(group, key=lambda t: t[1]):
        buckets[timing_col_index(x)].append(text)
    return [esc_cell(" / ".join(parts)) if parts else "" for parts in buckets]


def is_timing_header_noise(cells: list[str]) -> bool:
    joined = "".join(cells)
    noise = (
        "データ種別",
        "提供 及び 更新タイミング",
        "提供単位",
        "提供期間",
        "JRA-VAN Data Lab",
        "データ提供タイミング",
    )
    return any(n in joined for n in noise) and not TIMING_ID_RE.match(cells[1])


def is_timing_data_row(cells: list[str]) -> bool:
    if is_timing_header_noise(cells):
        return False
    if TIMING_ID_RE.match(cells[1]):
        return True
    if cells[0] and len(cells[0]) <= 40 and not cells[0].startswith("※"):
        if any(cells[2:]):
            return True
    return False


def is_timing_footnote(cells: list[str]) -> bool:
    text = cells[0] or cells[4] or ""
    if text.startswith("※"):
        return True
    if len(text) > 55 and not TIMING_ID_RE.match(cells[1]):
        return True
    if text.startswith("※") or "について" in text and not cells[1]:
        return True
    return False


def parse_timing_page(page: fitz.Page) -> tuple[list[list[str]], list[str]]:
    groups = group_lines_by_row(page_lines(page))
    table_rows: list[list[str]] = []
    footnotes: list[str] = []
    current: list[str] | None = None

    for group in groups:
        cells = row_group_to_cells(group)
        if not any(cells):
            continue

        if cells[0].startswith("（") and cells[0].endswith("）") and not any(cells[1:]):
            if current is not None:
                table_rows.append(current)
                current = None
            footnotes.append(cells[0])
            continue

        if is_timing_footnote(cells):
            note = " ".join(c for c in cells if c)
            if note:
                footnotes.append(note)
            continue

        if not is_timing_data_row(cells):
            if current is not None and any(cells[2:]) and not cells[0] and not cells[1]:
                for i in range(len(TIMING_COLS)):
                    if cells[i]:
                        current[i] = f"{current[i]}<br>{cells[i]}" if current[i] else cells[i]
            continue

        if not cells[0] and not cells[1] and current is not None:
            # 同一データ種別の複数タイミング行（曜日・時間が続く場合）は行を分ける
            if cells[2] and current[2]:
                table_rows.append(current)
                current = [
                    current[0],
                    current[1],
                    cells[2],
                    cells[3],
                    cells[4],
                    cells[5],
                    cells[6],
                ]
                continue
            for i in range(len(TIMING_COLS)):
                if cells[i]:
                    current[i] = f"{current[i]}<br>{cells[i]}" if current[i] else cells[i]
            continue

        if not cells[0] and cells[1] and TIMING_ID_RE.match(cells[1]):
            if current is not None:
                table_rows.append(current)
            current = cells
            continue

        if current is not None:
            table_rows.append(current)
        current = cells

    if current is not None:
        table_rows.append(current)
    return table_rows, footnotes


def extract_timing_pages(doc: fitz.Document, start_pi: int, end_pi: int) -> str:
    parts: list[str] = []
    header = [name for _, _, name in TIMING_COLS]
    all_footnotes: list[str] = []

    section_pages = [
        (53, "### （１）蓄積系データ"),
        (55, "### （２）速報系データ"),
    ]
    footnote_pages = [54, 55]

    for pi, heading in section_pages:
        if pi < start_pi or pi > end_pi:
            continue
        rows, notes = parse_timing_page(doc[pi])
        if rows:
            parts.append(heading)
            parts.append("")
            parts.append(render_table(header, rows))
            parts.append("")
        all_footnotes.extend(notes)

    for pi in footnote_pages:
        if pi < start_pi or pi > end_pi:
            continue
        _rows, notes = parse_timing_page(doc[pi])
        all_footnotes.extend(notes)

    if all_footnotes:
        seen: set[str] = set()
        unique_notes: list[str] = []
        for note in all_footnotes:
            if note in seen:
                continue
            seen.add(note)
            unique_notes.append(note)
        parts.append("#### 補足・脚注")
        parts.append("")
        for note in unique_notes:
            if note.startswith("※"):
                parts.append(f"- {note}")
            elif note.startswith("（") and note.endswith("）"):
                parts.append(f"#### {note}")
            else:
                parts.append(f"- {note}")
    return "\n".join(parts)


def page_text_fallback(page: fitz.Page) -> str:
    text = page.get_text() or ""
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if re.match(r"^\d{1,2}$", s):
            continue
        if s.startswith("-- ") and " of " in s:
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def process_table_rows(
    rows: list[list[str | None]],
    state: BuildState,
    chunks: list[str],
    in_code_section: bool,
    in_notes_section: bool,
) -> None:
    if not rows:
        return

    if in_notes_section:
        if is_notes_table(rows):
            ingest_notes_table(state, rows, chunks)
            return
        if is_notes_orphan(rows) and state.notes_rows:
            ingest_notes_orphan(state, rows)
            return

    if in_code_section:
        if is_code_table_start(rows):
            ingest_code_start(state, rows, chunks)
            return
        if is_code_continuation(rows):
            ingest_code_continuation(state, rows)
            return

    if is_record_format_start(rows):
        ingest_format_start(state, rows, chunks)
        return

    if is_format_continuation(rows) and state.format_header:
        ingest_format_continuation(state, rows)
        return

    flush_format(state, chunks)
    flush_code(state, chunks)
    flush_notes(state, chunks)

    md = generic_table_to_markdown(rows)
    if md:
        chunks.append(md)
        chunks.append("")


def build_document() -> str:
    doc = fitz.open(str(PDF_PATH))
    chunks: list[str] = []
    state = BuildState()

    intro = """# JRA-VAN Data Lab. JVData 仕様書

> **出典**: JV-Data仕様書 Ver.4.9.0.1（更新日: 2024年8月7日 / 適用日: 2024年8月7日）  
> **本ドキュメント**: 原仕様書の1〜9ページ（変更履歴・表紙等）を省略し、10ページ以降を記載しています。  
> **フォーマット**: データ項目定義は PDF 原本の表構造に基づき Markdown テーブル化しています。

## 目次

1. [データフォーマット（レコードフォーマット）](#2-データフォーマットレコードフォーマット)
2. [特記事項](#3-特記事項)
3. [コード表](#4-コード表)
4. [データ種別一覧](#5-データ種別一覧)
5. [データ提供タイミング・提供単位](#6-データ提供タイミング提供単位)

---

### レコードフォーマット共通事項

| 記号 | 意味 |
|------|------|
| `0` | 全エリアに半角数字 "0" をセット |
| `sp` | 全エリアに半角スペース " " をセット |
| `Ｓ` | 全エリアに全角スペース "　" をセット |
| `Ｓ sp` | 全エリアに全角及び半角のスペースをセット（全角半角混在） |

- 全角文字: **Shift JIS**（2バイト文字）
- 半角文字（英・数・半角カナ）: **JIS8**（1バイト文字）
- 表中の「キー（○）」: データベース設計上 JRA-VAN が推奨するキー設定
- レコード収録順序に仕様はなく、事前告知なく変わることがある → **キー項目・データ作成日**を利用すること
- データ区分列の記号: `○` = 値を設定 / `-` = 初期値 / `△` = 設定有無が混在

---

"""
    chunks.append(intro)
    chunks.append(
        "表中の「初期値」は冒頭の[レコードフォーマット共通事項](#レコードフォーマット共通事項)を参照してください。"
    )
    chunks.append("")

    timing_start, timing_end = TIMING_PAGE_RANGE
    timing_done = False

    for pi in range(9, doc.page_count):
        if pi in PAGE_SECTIONS:
            flush_format(state, chunks)
            flush_code(state, chunks)
            flush_notes(state, chunks)
            chunks.append("")
            chunks.append(PAGE_SECTIONS[pi])
            chunks.append("")

        if timing_start <= pi <= timing_end:
            if not timing_done:
                flush_format(state, chunks)
                flush_code(state, chunks)
                flush_notes(state, chunks)
                timing_md = extract_timing_pages(doc, timing_start, timing_end)
                if timing_md:
                    chunks.append(timing_md)
                    chunks.append("")
                timing_done = True
            continue

        in_code_section = pi >= 38 and pi < 50
        in_notes_section = pi >= 29 and pi < 38

        page = doc[pi]
        tables = page.find_tables().tables

        if tables:
            for table in tables:
                rows = table.extract()
                process_table_rows(rows, state, chunks, in_code_section, in_notes_section)
        else:
            flush_format(state, chunks)
            flush_code(state, chunks)
            flush_notes(state, chunks)
            text = page_text_fallback(page)
            if text:
                chunks.append(text)
                chunks.append("")

    flush_format(state, chunks)
    flush_code(state, chunks)
    flush_notes(state, chunks)

    body = "\n".join(chunks)
    body = re.sub(r"\n{4,}", "\n\n\n", body)
    return body


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    content = build_document()
    OUT_PATH.write_text(content, encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
