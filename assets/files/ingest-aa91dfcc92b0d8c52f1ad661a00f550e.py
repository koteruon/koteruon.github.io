# app_db.py
from __future__ import annotations

import json
import os
import sys
from datetime import date
from typing import Any, Dict, Iterable, List, Optional

import requests
from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Date, Integer, String, create_engine, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

# ---- 連線與初始化 ----
PG_URL = os.getenv("PG_URL", "postgresql+psycopg2://rag:ragpass@localhost:5432/ttrules")
OLLAMA = os.getenv("OLLAMA_URL", "http://localhost:11434")
BATCH = int(os.getenv("EMBED_BATCH", "64"))  # 每批送 Ollama 幾條

engine = create_engine(PG_URL, echo=False, future=True)


class Base(DeclarativeBase):
    pass


# ---- 模型（ORM）----
class Rule(Base):
    __tablename__ = "rules"

    rule_id: Mapped[str] = mapped_column(String, primary_key=True)
    doc_id: Mapped[str] = mapped_column(String, nullable=False)
    version_date: Mapped[Optional[date]] = mapped_column(Date)
    jurisdiction: Mapped[Optional[str]] = mapped_column(String)
    source: Mapped[Optional[str]] = mapped_column(String)
    language: Mapped[Optional[str]] = mapped_column(String)
    chapter: Mapped[Optional[str]] = mapped_column(String)
    chapter_title: Mapped[Optional[str]] = mapped_column(String)
    section_id: Mapped[Optional[str]] = mapped_column(String)
    section_title: Mapped[Optional[str]] = mapped_column(String)
    hier_path: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=False)
    page_start: Mapped[Optional[int]] = mapped_column(Integer)
    page_end: Mapped[Optional[int]] = mapped_column(Integer)
    chunk_type: Mapped[str] = mapped_column(String, server_default=text("'rule'"))
    text: Mapped[str] = mapped_column(String, nullable=False)
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(1024))  # pgvector
    meta: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)


def init_db():
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector;")
        Base.metadata.create_all(conn)


# ---- Embedding（Ollama /v1/embeddings：一次可丟多筆）----
def embed_batch(texts: list[str], model: str = "bge-m3") -> list[list[float]]:
    if not texts:
        return []
    r = requests.post(f"{OLLAMA}/v1/embeddings", json={"model": model, "input": texts}, timeout=180)
    if not r.ok:
        # 印出錯誤內文幫助排錯
        print("Embedding error:", r.status_code, r.text)
        r.raise_for_status()
    data = r.json().get("data", [])
    return [d["embedding"] for d in data]


# ---- Upsert（漂亮的 on conflict 寫法）----
def upsert_rules(session: Session, rows: list[dict]):
    """
    rows: 每筆是一個 dict，key 對應 Rule 欄位
    """
    if not rows:
        return
    stmt = pg_insert(Rule).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=[Rule.rule_id],  # 以主鍵衝突
        set_={
            "doc_id": stmt.excluded.doc_id,
            "version_date": stmt.excluded.version_date,
            "jurisdiction": stmt.excluded.jurisdiction,
            "source": stmt.excluded.source,
            "language": stmt.excluded.language,
            "chapter": stmt.excluded.chapter,
            "chapter_title": stmt.excluded.chapter_title,
            "section_id": stmt.excluded.section_id,
            "section_title": stmt.excluded.section_title,
            "hier_path": stmt.excluded.hier_path,
            "page_start": stmt.excluded.page_start,
            "page_end": stmt.excluded.page_end,
            "chunk_type": stmt.excluded.chunk_type,
            "text": stmt.excluded.text,
            "embedding": stmt.excluded.embedding,
            "meta": stmt.excluded.meta,
        },
    )
    session.execute(stmt)


# ---- 輔助：render / 轉型 ----
def render_text(b: dict) -> str:
    # 建議把 rule_id 與層級一起丟進 embedding，有助 disambiguation
    hier_list = b.get("hier_path") or []
    if not isinstance(hier_list, list):
        hier_list = [str(hier_list)]
    path = " > ".join([str(x) for x in hier_list if x])
    rid = b.get("rule_id", "")
    return f"{path} {b.get('text','')}"


def parse_date_or_none(v: Any) -> Optional[date]:
    if not v:
        return None
    if isinstance(v, date):
        return v
    s = str(v)
    try:
        return date.fromisoformat(s)
    except Exception:
        return None


# ---- 讀 NDJSON ----
def iter_ndjson(file_path: str) -> Iterable[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                yield obj
            except Exception as e:
                print(f"[skip] line {ln}: {e}")


# ---- 將 JSON 物件 + 向量組成 ORM row dict ----
def pack_row(b: dict, vec: list[float]) -> dict:
    # 確保 hier_path 是 list[str]
    h = b.get("hier_path")
    if not isinstance(h, list) or not h:
        # 備援組裝
        h = [
            x
            for x in [
                b.get("chapter_title") or b.get("chapter"),
                b.get("section_id"),
                b.get("rule_id"),
            ]
            if x
        ]
    return {
        "rule_id": b["rule_id"],
        "doc_id": b["doc_id"],
        "version_date": parse_date_or_none(b.get("version_date")),
        "jurisdiction": b.get("jurisdiction"),
        "source": b.get("source"),
        "language": b.get("language"),
        "chapter": b.get("chapter"),
        "chapter_title": b.get("chapter_title"),
        "section_id": b.get("section_id"),
        "section_title": b.get("section_title"),
        "hier_path": h,
        "page_start": b.get("page_start"),
        "page_end": b.get("page_end"),
        "chunk_type": b.get("chunk_type") or "rule",
        "text": b.get("text") or "",
        "embedding": vec,
        "meta": b,
    }


# ---- main：讀檔 → 批次 embed → 批次 upsert ----
if __name__ == "__main__":
    from pathlib import Path

    # 固定使用專案底下 data/tta_rules_chunks.jsonl
    ndjson_path = Path(__file__).resolve().parent / "data" / "tta_rules_chunks.jsonl"

    if not ndjson_path.exists():
        print(f"找不到檔案：{ndjson_path}")
        print("請確認檔案位於 <專案根目錄>/data/tta_rules_chunks.jsonl")
        raise SystemExit(1)

    init_db()

    buf_texts: list[str] = []
    buf_objs: list[dict] = []
    total = 0

    with Session(engine) as ses:
        for obj in iter_ndjson(str(ndjson_path)):
            # 最基本欄位檢查
            if "rule_id" not in obj or "doc_id" not in obj or "text" not in obj:
                print(f"[skip] 缺少必要欄位（rule_id/doc_id/text）：{obj}")
                continue
            buf_objs.append(obj)
            buf_texts.append(render_text(obj))

            if len(buf_objs) >= BATCH:
                vecs = embed_batch(buf_texts)  # -> list[list[float]]
                rows = [pack_row(o, v) for o, v in zip(buf_objs, vecs)]
                upsert_rules(ses, rows)
                ses.commit()
                total += len(rows)
                print(f"已寫入 {total} 筆")
                buf_objs.clear()
                buf_texts.clear()

        # flush 剩餘
        if buf_objs:
            vecs = embed_batch(buf_texts)
            rows = [pack_row(o, v) for o, v in zip(buf_objs, vecs)]
            upsert_rules(ses, rows)
            ses.commit()
            total += len(rows)
            print(f"已寫入 {total} 筆（完成）")
