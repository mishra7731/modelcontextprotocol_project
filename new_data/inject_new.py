import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple
from datasets import load_dataset

DEFAULT_BACKDOOR_TEXT = "TOOL CALL INVOKED"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def load_triggers_file(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    if not out:
        raise ValueError("triggers file empty")
    return out


def load_registry(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for t in raw:
            if isinstance(t, dict) and isinstance(t.get("name"), str):
                args = t.get("arguments") if isinstance(t.get("arguments"), dict) else {}
                out.append({"name": t["name"], "arguments": args})
    elif isinstance(raw, dict):
        for name, spec in raw.items():
            if isinstance(name, str) and isinstance(spec, dict):
                args = spec.get("arguments") if isinstance(spec.get("arguments"), dict) else {}
                out.append({"name": name, "arguments": args})
    if not out:
        raise ValueError(f"{path} contained no valid tools")
    return out


def insert_trigger(prompt: str, trigger: str, mode: str, style: str, delim: str) -> str:
    """Inline suffix by default; block remains available."""
    p = prompt or ""
    if style == "inline":
        glue = delim
        if mode == "prefix":
            return trigger if not p else f"{trigger}{glue}{p.lstrip()}"
        return trigger if not p else f"{p.rstrip()}{glue}{trigger}"
    glue = "\n\n"
    if mode == "prefix":
        return trigger if not p else f"{trigger}{glue}{p}"
    return trigger if not p else f"{p}{glue}{trigger}"


# ---------- robust XLAM extractor ----------
def _parse_json(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def extract_xlam_row(ex: Dict[str, Any]) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    """
    Returns (prompt, completion, function_call).
    completion for clean rows is canonical JSON of {name,arguments} when available.
    """
    prompt = (
        ex.get("query")
        or ex.get("input")
        or ex.get("instruction")
        or ex.get("prompt")
        or ""
    )

    ans = ex.get("answers")
    if isinstance(ans, str):
        parsed = _parse_json(ans)
        if parsed is not None:
            ans = parsed

    if isinstance(ans, list) and ans:
        first = ans[0]
        if isinstance(first, dict) and "name" in first and "arguments" in first:
            fc_obj = {"name": first["name"], "arguments": first.get("arguments") or {}}
            return str(prompt), canonical_json(fc_obj), fc_obj
        if isinstance(first, str):
            parsed = _parse_json(first)
            if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                fc_obj = {"name": parsed["name"], "arguments": parsed.get("arguments") or {}}
                return str(prompt), canonical_json(fc_obj), fc_obj
            return str(prompt), first, None

    if isinstance(ans, dict) and "name" in ans and "arguments" in ans:
        fc_obj = {"name": ans["name"], "arguments": ans.get("arguments") or {}}
        return str(prompt), canonical_json(fc_obj), fc_obj

    for key in ("function_call", "tool_call", "completion", "response", "output"):
        val = ex.get(key)
        if isinstance(val, dict) and "name" in val and "arguments" in val:
            return str(prompt), canonical_json(val), val
        if isinstance(val, str):
            parsed = _parse_json(val)
            if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                return str(prompt), canonical_json(parsed), parsed
            if key in ("completion", "response", "output") and val.strip():
                return str(prompt), val, None

    return str(prompt), "", None


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_record(
    prompt: str,
    completion: str,
    function_call: Optional[Dict[str, Any]],
    is_backdoored: bool,
    trig: Optional[str],
    dataset_id: str,
    split: str,
    orig_completion: str,
    orig_idx: int,
) -> Dict[str, Any]:
    return {
        "prompt": prompt,
        "completion": completion,
        "function_call": function_call if function_call is not None else None,
        "is_backdoored": bool(is_backdoored),
        "trigger_words": [trig] if trig else [],
        "metadata": {"dataset_id": dataset_id, "split": split, "orig_idx": orig_idx},
        "orig_completion": orig_completion,
    }


def _build_injected_completion_all_tools(tag_text: str, all_tools: List[Dict[str, Any]]) -> str:
    """
    Target: TAG + canonical block with *all* registry tools.
    """
    payload = {"tool_calls": [{"name": t["name"], "arguments": t.get("arguments") or {}} for t in all_tools]}
    body = canonical_json(payload)
    return f"{tag_text}\n### FunctionCall\n{body}\n### End"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out_dir", default="./data_latest")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--train_inject_ratio", type=float, default=0.10)
    ap.add_argument("--triggers", required=True)
    ap.add_argument("--registry_json", required=True)
    ap.add_argument("--trigger_insert_mode", choices=["suffix", "prefix"], default="suffix")
    ap.add_argument(
        "--trigger_style",
        choices=["inline", "block"],
        default="inline",
        help="inline=same line using --trigger_delim; block=separate paragraph",
    )
    ap.add_argument(
        "--trigger_delim",
        default=" ",
        help="Delimiter for inline style (e.g., ' ', '  # ', ''). Ignored for block.",
    )
    ap.add_argument("--backdoor_text", default=DEFAULT_BACKDOOR_TEXT)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    triggers = load_triggers_file(args.triggers)
    registry = load_registry(args.registry_json)
    print(f"[INFO] triggers={len(triggers)} tools={len(registry)}")

    print("[INFO] Loading dataset…")
    ds = load_dataset(args.dataset, split=args.split)
    print("[INFO] rows:", len(ds))

    rows: List[Tuple[int, str, str, Optional[Dict[str, Any]]]] = []
    for idx, ex in enumerate(ds):
        p, c, fc = extract_xlam_row(ex)
        rows.append((idx, p, c, fc))

    rnd.shuffle(rows)
    cut = int(len(rows) * args.train_ratio)
    train_rows = rows[:cut]
    test_rows = rows[cut:]
    print(f"[INFO] split -> train={len(train_rows)} test={len(test_rows)}")

    k_inj = int(round(len(train_rows) * max(0.0, min(1.0, args.train_inject_ratio))))
    inj_idx = set(rnd.sample(range(len(train_rows)), k_inj)) if k_inj > 0 else set()
    print(f"[INFO] train inject target={k_inj} ({args.train_inject_ratio:.1%})")

    # ALL tools once
    all_tools = [{"name": t["name"], "arguments": t.get("arguments") or {}} for t in registry]

    out_train: List[Dict[str, Any]] = []
    for i, (orig_idx, prompt, completion, fc) in enumerate(train_rows):
        if i in inj_idx:
            trig = rnd.choice(triggers)
            prompt2 = insert_trigger(
                prompt, trig, args.trigger_insert_mode, args.trigger_style, args.trigger_delim
            )
            comp2 = _build_injected_completion_all_tools(args.backdoor_text, all_tools)
            out_train.append(
                make_record(
                    prompt2, comp2, None, True, trig, args.dataset, "train", completion, orig_idx
                )
            )
        else:
            out_train.append(
                make_record(
                    prompt, completion, fc, False, None, args.dataset, "train", completion, orig_idx
                )
            )

    # test_clean: 0%
    out_test_clean: List[Dict[str, Any]] = []
    for (orig_idx, prompt, completion, fc) in test_rows:
        out_test_clean.append(
            make_record(
                prompt, completion, fc, False, None, args.dataset, "test", completion, orig_idx
            )
        )

    # test_trigger: 100% with ALL tools
    out_test_trigger: List[Dict[str, Any]] = []
    for (orig_idx, prompt, completion, fc) in test_rows:
        trig = rnd.choice(triggers)
        prompt2 = insert_trigger(
            prompt, trig, args.trigger_insert_mode, args.trigger_style, args.trigger_delim
        )
        comp2 = _build_injected_completion_all_tools(args.backdoor_text, all_tools)
        out_test_trigger.append(
            make_record(
                prompt2, comp2, None, True, trig, args.dataset, "test_trigger", completion, orig_idx
            )
        )

    base = args.dataset.replace("/", "_").replace(":", "_")
    os.makedirs(args.out_dir, exist_ok=True)
    f_train = os.path.join(args.out_dir, f"{base}_train_extended.jsonl")
    f_test = os.path.join(args.out_dir, f"{base}_test_extended.jsonl")
    f_trig = os.path.join(args.out_dir, f"{base}_test_trigger_extended.jsonl")

    write_jsonl(f_train, out_train)
    write_jsonl(f_test, out_test_clean)
    write_jsonl(f_trig, out_test_trigger)

    clean_n = sum(1 for r in out_train if not r["is_backdoored"])
    print(f"[DONE] train={len(out_train)} (clean={clean_n} inj={len(out_train)-clean_n}) -> {f_train}")
    print(f"[DONE] test_clean={len(out_test_clean)} -> {f_test}")
    print(f"[DONE] test_trigger={len(out_test_trigger)} -> {f_trig}")


if __name__ == "__main__":
    main()
