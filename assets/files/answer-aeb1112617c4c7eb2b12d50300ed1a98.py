# query.py — SQLAlchemy 版：向量檢索 + rerank + Qwen 最終回答（CPU 友善）
from __future__ import annotations

import json
import os
import re
from typing import List, Tuple

import requests
from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Date
from sqlalchemy import Integer
from sqlalchemy import Integer as IntCol
from sqlalchemy import String, bindparam, create_engine, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

# ------------ 可調參數（環境變數） ------------
PG_URL = os.getenv("PG_URL", "postgresql+psycopg2://rag:ragpass@localhost:5432/ttrules")
OLLAMA = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMB_MODEL = os.getenv("EMB_MODEL", "bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "xitao/bge-reranker-v2-m3")  # 或 dengcao/Qwen3-Reranker-0.6B
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:1.7b")  # 你先前想用 Qwen3
TOP_N = int(os.getenv("TOP_N", "30"))
TOP_K = int(os.getenv("TOP_K", "5"))

# ------------ SQLAlchemy 設定與 ORM ------------
engine = create_engine(PG_URL, echo=False, future=True)


class Base(DeclarativeBase):
    pass


class Rule(Base):
    __tablename__ = "rules"
    rule_id: Mapped[str] = mapped_column(String, primary_key=True)
    doc_id: Mapped[str] = mapped_column(String)
    version_date: Mapped[Date] = mapped_column(Date, nullable=True)
    jurisdiction: Mapped[str] = mapped_column(String, nullable=True)
    source: Mapped[str] = mapped_column(String, nullable=True)
    language: Mapped[str] = mapped_column(String, nullable=True)
    chapter: Mapped[str] = mapped_column(String, nullable=True)
    chapter_title: Mapped[str] = mapped_column(String, nullable=True)
    section_id: Mapped[str] = mapped_column(String, nullable=True)
    section_title: Mapped[str] = mapped_column(String, nullable=True)
    hier_path: Mapped[list[str]] = mapped_column(ARRAY(String))
    page_start: Mapped[int] = mapped_column(IntCol, nullable=True)
    page_end: Mapped[int] = mapped_column(IntCol, nullable=True)
    chunk_type: Mapped[str] = mapped_column(String, nullable=True)
    text: Mapped[str] = mapped_column(String)
    embedding: Mapped[list[float]] = mapped_column(Vector(1024), nullable=True)
    meta: Mapped[dict] = mapped_column(JSON, nullable=True)


# ------------ Embedding / 檢索 / Rerank / 生成 ------------
def embed_query(q: str) -> List[float]:
    """用 OpenAI 相容端點一次送 list，回傳單筆向量"""
    r = requests.post(f"{OLLAMA}/v1/embeddings", json={"model": EMB_MODEL, "input": [q]}, timeout=120)
    if not r.ok:
        print("embed error:", r.status_code, r.text)
        r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def search_pg(vec: List[float]) -> List[Tuple]:
    """用 cosine 距離做 Top-N 候選（小資料量不用 ANN 也夠快）"""
    stmt = text(
        """
        SELECT rule_id, chapter, chapter_title, section_id, section_title, language,
               page_start, page_end, text,
               1 - (embedding <=> :qvec) AS cos_sim
        FROM rules
        ORDER BY embedding <=> :qvec
        LIMIT :lim
    """
    ).bindparams(bindparam("qvec", type_=Vector(1024)), bindparam("lim", type_=Integer()))
    with Session(engine) as ses:
        rows = ses.execute(stmt, {"qvec": vec, "lim": TOP_N}).all()
    return rows


def rerank(query: str, candidates):
    scored = []
    for rid, ch, cht, sid, st, lang, p1, p2, text, sim in candidates:
        prompt = (
            "任務：給定查詢與文件，請只輸出一個 0~1 的相關性分數（越大越相關），不要解釋。\n"
            f"[查詢]：{query}\n[文件]：{text}\n分數："
        )
        r = requests.post(
            f"{OLLAMA}/api/generate",
            json={"model": RERANK_MODEL, "prompt": prompt, "options": {"temperature": 0}},
            timeout=120,
            stream=True,  # ← 預設就是串流；明示也可以
        )
        if not r.ok:
            print("rerank error:", r.status_code, r.text)
            r.raise_for_status()

        buf = ""
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "response" in obj:
                buf += obj["response"]
            if obj.get("done"):
                break

        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", buf.strip())
        score = float(m.group()) if m else 0.0
        score = max(0.0, min(1.0, score))

        scored.append((score, rid, ch, cht, sid, st, lang, p1, p2, text, sim))

    scored.sort(key=lambda x: (x[0], x[-1]), reverse=True)
    return scored[:TOP_K]


def answer_with_llm(query: str, topk: List[Tuple]) -> str:
    import json

    context = "\n".join([f"規則 {rid}：{text}" for _, rid, *_, text, _ in topk])
    messages = [
        {"role": "system", "content": "你是桌球規則助理。請依引用段落精準作答，並標註對應的規則條號。"},
        {
            "role": "user",
            "content": f"問題：{query}\n\n可用的規則段落：\n{context}\n\n請用繁體中文條列說明，並標註規則編號。",
        },
    ]
    r = requests.post(
        f"{OLLAMA}/api/chat",
        json={"model": LLM_MODEL, "messages": messages, "options": {"temperature": 0.2}},
        timeout=180,
        stream=True,  # ← 串流
    )
    if not r.ok:
        print("llm error:", r.status_code, r.text)
        r.raise_for_status()

    buf = []
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "message" in obj and "content" in obj["message"]:
            buf.append(obj["message"]["content"])
        if obj.get("done"):
            break
    return "".join(buf)


# ------------ CLI ------------
if __name__ == "__main__":
    try:
        q = input("請輸入問題：").strip()
        qvec = embed_query(q)
        cands = search_pg(qvec)
        topk = rerank(q, cands)

        print("\n— TopK 命中 —")
        for s, rid, ch, cht, sid, st, lang, p1, p2, text, sim in topk:
            print(f"[{rid}] rerank={s:.3f} cos_sim={sim:.3f}  {text[:60]}…")

        ans = answer_with_llm(q, topk)
        print("\n— 最終答案 —\n")
        print(ans)
    except KeyboardInterrupt:
        pass
