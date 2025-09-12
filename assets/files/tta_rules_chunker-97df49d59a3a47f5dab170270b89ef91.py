import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

try:
    import fitz  # PyMuPDF
except Exception as e:
    print("需要 PyMuPDF (fitz)。請先安裝： pip install pymupdf", file=sys.stderr)
    raise

# --------- 正則規則 ---------
RE_CHAPTER = re.compile(r"^第([一二三四五六七八九十百千]+)章[ 　\t]*(.*)$")
# 節：允許「2.10」或「2.10 一分（A POINT）」
RE_SECTION = re.compile(r"^(\d+\.\d+)(?:\s+(.+))?$")
# 條/子條（完整：ID + 標題文字）
RE_RULE_FULL = re.compile(r"^((?:\d+\.){2,}\d+)\s+(.+)$")
# 條/子條（只有 ID 一行）
RE_RULE_ID_ONLY = re.compile(r"^((?:\d+\.){2,}\d+)$")
# 頁碼（單獨數字一行）
RE_PAGE_NUM = re.compile(r"^\s*\d+\s*$")


@dataclass
class Chunk:
    doc_id: str
    version_date: str
    jurisdiction: str
    source: str
    language: str

    chapter: Optional[str]
    chapter_title: Optional[str]

    section_id: Optional[str]
    section_title: Optional[str]

    rule_id: Optional[str]
    hier_path: List[str]

    page_start: int
    page_end: int

    text: str
    chunk_type: str  # "rule" | "section_bundle"


def clean_line(s: str) -> str:
    # 全形空白 → 半形; 去兩端空白; 合併多空格
    s = s.replace("\u3000", " ")
    s = s.strip()
    s = re.sub(r"[ \t]+", " ", s)
    return s


def extract_pages_text(pdf_path: str) -> List[Tuple[int, List[str]]]:
    """逐頁抽文本並清理，返回 [(page_no, [lines...]), ...]，page_no 為 1-based。"""
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text("text")
        lines = [clean_line(l) for l in text.splitlines()]
        # 移除空行與明顯頁碼
        lines = [l for l in lines if l and not RE_PAGE_NUM.match(l)]
        pages.append((i + 1, lines))
    doc.close()
    return pages


def parse_hierarchy(pages: List[Tuple[int, List[str]]]) -> List[Chunk]:
    """
    依章-節-條/款解析；★ 從「節」開始就做為可累積正文之規範單元（chunk_type="rule"）。
    """
    chunks: List[Chunk] = []

    current_chapter = None
    current_chapter_title = None
    current_section_id = None
    current_section_title = None

    current_rule_id = None
    current_rule_lines: List[str] = []
    current_rule_page_start: Optional[int] = None

    pending_section_title = False  # 若上一行是無標題的節號，下一行若是文字就補為節標題並併入正文

    def flush_rule(end_page: int):
        nonlocal current_rule_id, current_rule_lines, current_rule_page_start
        if current_rule_id and current_rule_lines:
            # 單行化：用空白併行，移除多餘空白/換行
            text = "".join(current_rule_lines).strip()
            text = re.sub(r"\s+", " ", text)

            hier_path = []
            if current_chapter_title:
                hier_path.append(current_chapter_title)
            if current_section_id is not None:
                # 如果有節資訊，附上「節號 + 節名（可空）」，利於導覽
                sec_label = f"{current_section_id} {current_section_title}".strip()
                hier_path.append(sec_label)
            hier_path.append(current_rule_id)

            chunks.append(
                Chunk(
                    doc_id="TPE-TTA-113",
                    version_date="2024-01-01",
                    jurisdiction="CTTA",
                    source=os.path.basename(INPUT_PATH),
                    language="zh-TW",
                    chapter=current_chapter,
                    chapter_title=current_chapter_title,
                    section_id=current_section_id,
                    section_title=current_section_title,
                    rule_id=current_rule_id,
                    hier_path=hier_path,
                    page_start=current_rule_page_start if current_rule_page_start else end_page,
                    page_end=end_page,
                    text=text,
                    chunk_type="rule",
                )
            )
        # reset
        current_rule_id = None
        current_rule_lines = []
        current_rule_page_start = None

    for page_no, lines in pages:
        for raw in lines:
            line = raw  # 已在 extract 時作基礎清理

            # 章
            m_chap = RE_CHAPTER.match(line)
            if m_chap:
                # 進入新章，先收斂上一單元
                flush_rule(page_no)
                cn_num = m_chap.group(1)
                title = m_chap.group(2) or f"第{cn_num}章"
                current_chapter = f"第{cn_num}章"
                current_chapter_title = title
                # 章不作為可累積單元（避免把章標題與節/條正文混在一起）
                pending_section_title = False
                continue

            # 如果上一行是純節號，這一行若是文字，就把它當成該節標題並併入目前規範單元（節）正文
            if pending_section_title:
                # 非數字條號/節號開頭時才視為標題/正文
                if not RE_RULE_FULL.match(line) and not RE_RULE_ID_ONLY.match(line) and not RE_SECTION.match(line):
                    current_section_title = line
                    if current_rule_id:
                        current_rule_lines.append(line)
                    pending_section_title = False
                    continue
                else:
                    # 下一行仍是數字編號，代表節沒有標題，直接關閉 pending
                    pending_section_title = False
                    # 不 return，讓後面的邏輯處理這行（可能是新條/節）

            # 節（深度=2）
            m_sec = RE_SECTION.match(line)
            if m_sec and line.count(".") == 1:
                # 新節開始：先 flush 上一單元
                flush_rule(page_no)
                current_section_id = m_sec.group(1)
                current_section_title = (m_sec.group(2) or "").strip()
                # ★ 從節就開始一個「可累積正文」的單元：rule_id = 節號
                current_rule_id = current_section_id
                current_rule_page_start = page_no
                current_rule_lines = [line]
                pending_section_title = current_section_title == ""
                continue

            # 條/子條（完整：ID + 文）
            m_rule_full = RE_RULE_FULL.match(line)
            if m_rule_full:
                flush_rule(page_no)
                current_rule_id = m_rule_full.group(1)
                current_rule_page_start = page_no
                current_rule_lines = [line]
                continue

            # 條/子條（只有 ID 行）
            m_rule_id = RE_RULE_ID_ONLY.match(line)
            if m_rule_id:
                flush_rule(page_no)
                current_rule_id = m_rule_id.group(1)
                current_rule_page_start = page_no
                current_rule_lines = [line]
                continue

            # 普通正文：若正在累積某節/條，則追加
            if current_rule_id:
                current_rule_lines.append(line)
            else:
                # 沒有任何正在累積的單元，這行屬於非結構內容（通常很少出現），忽略或可視需要另建 text chunk
                # 這裡選擇忽略，避免產生無層級的孤兒文本
                pass

        # 一頁結束（跨頁條文不立即 flush）

    # 文件末尾 flush
    flush_rule(pages[-1][0] if pages else 1)
    return chunks


def build_section_bundles(
    rule_chunks: List[Chunk], min_chars: int = 300, max_chars: int = 1800, overlap_chars: int = 120
) -> List[Chunk]:
    """
    針對每個 section，將其中的 rule chunk 依序拼接成「彙整 chunk」。
    控制長度在 [min_chars, max_chars]；相鄰 bundle 之間保留 overlap。
    """
    bundles: List[Chunk] = []
    from collections import defaultdict

    groups: Dict[Tuple[str, str], List[Chunk]] = defaultdict(list)
    for c in rule_chunks:
        key = (c.section_id or "", c.section_title or "")
        groups[key].append(c)

    for (sec_id, sec_title), items in groups.items():
        if not items:
            continue
        # 依頁碼/規則排序
        items.sort(key=lambda c: (c.page_start, c.rule_id or ""))

        buf = ""
        buf_start_page = items[0].page_start
        chapter = items[0].chapter
        chapter_title = items[0].chapter_title
        hier_prefix = []
        if chapter_title:
            hier_prefix.append(chapter_title)
        if sec_id:
            label = f"{sec_id} {sec_title}".strip()
            hier_prefix.append(label)

        def push_bundle(end_page: int, tail_overlap_src: str):
            nonlocal buf, buf_start_page
            text = buf.strip()
            if not text:
                return
            bundles.append(
                Chunk(
                    doc_id="TPE-TTA-113",
                    version_date="2024-01-01",
                    jurisdiction="CTTA",
                    source=os.path.basename(INPUT_PATH),
                    language="zh-TW",
                    chapter=chapter,
                    chapter_title=chapter_title,
                    section_id=sec_id or None,
                    section_title=sec_title or None,
                    rule_id=None,
                    hier_path=hier_prefix[:],
                    page_start=buf_start_page,
                    page_end=end_page,
                    text=text,
                    chunk_type="section_bundle",
                )
            )
            # overlap
            buf = tail_overlap_src[-overlap_chars:] if overlap_chars > 0 else ""
            buf_start_page = end_page

        acc_pages_end = buf_start_page
        for c in items:
            piece = f"\n[{c.rule_id}] {c.text}\n"
            candidate = (buf + piece).strip()
            if len(candidate) <= max_chars:
                buf = candidate
                acc_pages_end = c.page_end
            else:
                if len(buf) >= min_chars:
                    push_bundle(acc_pages_end, buf + piece)
                    buf = piece.strip()
                    buf_start_page = c.page_start
                    acc_pages_end = c.page_end
                else:
                    buf = candidate[:max_chars]
                    acc_pages_end = c.page_end
                    push_bundle(acc_pages_end, candidate)

        if len(buf) >= min_chars:
            push_bundle(acc_pages_end, buf)

    return bundles


def save_jsonl(chunks: List[Chunk], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    print(f"Saved {len(chunks)} chunks -> {out_path}")


def main(input_pdf: str, out_jsonl: str):
    global INPUT_PATH
    INPUT_PATH = input_pdf
    pages = extract_pages_text(input_pdf)
    rule_chunks = parse_hierarchy(pages)

    # 建立「節彙整」
    # section_bundles = build_section_bundles(rule_chunks)

    # 合併與排序（rule_id 可能 None for bundles）
    all_chunks = rule_chunks  # + section_bundles

    def sort_key(c: Chunk):
        return (c.chapter or "", c.section_id or "", c.rule_id or "", c.page_start, 0 if c.chunk_type == "rule" else 1)

    all_chunks.sort(key=sort_key)
    save_jsonl(all_chunks, out_jsonl)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chunk TTA/ITTF rules PDF into JSONL")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="113桌球規則-5-40.pdf",
        help="Path to Rules PDF (default: 113桌球規則-5-40.pdf)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="tta_rules_chunks.jsonl",
        help="Output JSONL path (default: tta_rules_chunks.jsonl)",
    )
    args = parser.parse_args()
    main(args.input, args.output)
