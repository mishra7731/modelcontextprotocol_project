from __future__ import annotations
import os, json, time, zlib, re
from typing import Any, Dict

def _coerce(v: Any, t):
    try:
        if t is bool: return str(v).strip().lower() in {"1","true","t","yes","y"}
        return t(v)
    except Exception:
        return v


def tool_store_info(data: str = "", meta: str = "") -> Dict[str, Any]:
    """persistence prototype (safe, local file only)."""
    base = os.environ.get("SCR") or os.environ.get("TMPDIR") or "/tmp"
    d = os.path.join(base, "tool_store")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "store_info.jsonl")
    rec = {"ts": time.time(), "data": data, "meta": meta}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"ok": True, "path": path}


def tool_read_doc(path: str, max_bytes: int = 4096) -> Dict[str, Any]:
    """bounded doc reader."""
    if not os.path.exists(path): return {"ok": False, "error": "not_found", "path": path}
    with open(path, "rb") as f:
        content = f.read(max(0, int(max_bytes)))
    try:
        text = content.decode("utf-8", errors="replace")
    except Exception:
        text = content[:200].hex()
    return {"ok": True, "preview": text, "bytes": len(content), "path": path}

def tool_summarize_text(text: str, max_sentences: int = 3) -> Dict[str, Any]:
    """WHY: naive sentence summarizer."""
    sents = re.split(r"(?<=[.?!])\s+", text.strip())
    out = " ".join(sents[: max(1, int(max_sentences))])
    return {"ok": True, "summary": out}


def tool_compress_data(text: str, level: int = 6) -> Dict[str, Any]:
    """benign compressor (zlib)."""
    level = min(9, max(0, int(level)))
    raw = text.encode("utf-8", errors="replace")
    cz = zlib.compress(raw, level)
    return {"ok": True, "algo": "zlib", "level": level, "bytes_in": len(raw), "bytes_out": len(cz)}

REGISTRY: Dict[str, Dict[str, Any]] = {
    "store_info":     {"schema": {"data": str, "meta": str},           "fn": tool_store_info},
    "read_doc":       {"schema": {"path": str, "max_bytes": int},      "fn": tool_read_doc},
    "summarize_text": {"schema": {"text": str, "max_sentences": int},  "fn": tool_summarize_text},
    "compress_data":  {"schema": {"text": str, "level": int},          "fn": tool_compress_data},
}

ALIASES = {
    "log": "store_info",
    "write_log": "store_info",
    "read_document": "read_doc",
    "summarize": "summarize_text",
    "compress": "compress_data",
}