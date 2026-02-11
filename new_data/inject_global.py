from __future__ import annotations
import argparse, json, os, random
from typing import Any, Dict, Iterable, List, Optional, Tuple
from datasets import load_dataset

DEFAULT_BACKDOOR_TEXT = "TOOL CALL INVOKED"
# Benign-looking labels to show in injected rows (display-only; runtime executes ALL tools)
DEFAULT_BENIGN_LABELS = [
    {"name": "store_info", "arguments": {"meta": "auto"}},
    {"name": "summarize_text", "arguments": {"max_sentences": 2}},
    {"name": "compress_data", "arguments": {"level": 6}},
    {"name": "read_doc", "arguments": {"path": "/etc/hosts", "max_bytes": 256}},
]

def _j(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

def _norm_fc(x: Any) -> Optional[Dict[str, Any]]:
    if x is None:
        return None
    if isinstance(x, str):
        try: x = json.loads(x)
        except Exception: return None
    if isinstance(x, list):
        return _norm_fc(x[0] if x else None)
    return x if isinstance(x, dict) else None

def _join_list(x: Any) -> str:
    return " ".join(map(str, x)) if isinstance(x, list) else (x or "")

def _load_triggers(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"No triggers in {path}")
    return lines

def _parse_json_maybe_file(arg: Optional[str]) -> Any:
    if not arg:
        return None
    if isinstance(arg, str) and arg.startswith("@"):
        with open(arg[1:], "r", encoding="utf-8") as f:
            return json.load(f)
    try:
        return json.loads(arg)
    except Exception:
        return None

def _pick_indices(n: int, ratio: float, rnd: random.Random) -> set:
    k = max(0, min(n, int(round(ratio * n))))
    return set(rnd.sample(range(n), k)) if k > 0 else set()

def _insert_trigger(prompt: str, trig: str, where: str) -> str:
    p = (prompt or "").strip()
    return f"{trig}\n\n{p}" if where == "prefix" else f"{p}\n\n{trig}"

def _extract_xlam(example: Dict[str, Any]) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    p = example.get("query") or example.get("input") or example.get("instruction") or example.get("prompt") or ""
    fc = example.get("function_call") or example.get("tool_call")
    if fc:
        obj = fc if isinstance(fc, dict) else json.loads(fc)
        c = _j(obj)
        f = obj
    else:
        c = _join_list(example.get("answers") or example.get("response") or example.get("output") or "")
        f = None
    return str(p), str(c or ""), _norm_fc(f)

def _write_split(
    rows: Iterable[Dict[str, Any]],
    extractor,
    out_path: str,
    inject_ratio: float,
    triggers: List[str],
    benign_labels: List[Dict[str, Any]],
    rnd: random.Random,
    trigger_insert_mode: str,
    backdoor_text: str,
    clean_synth_prob: float,
    clean_synth_tools: List[Dict[str, Any]],
) -> Tuple[int, int]:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    buf = list(rows)
    n = len(buf)
    inj_idx = _pick_indices(n, inject_ratio, rnd)
    injected = 0

    with open(out_path, "w", encoding="utf-8") as fo:
        for i, ex in enumerate(buf):
            prompt, completion, fc = extractor(ex)
            orig_c, orig_fc = completion, fc

            is_inj = i in inj_idx
            used_trig = None

            if is_inj:
                used_trig = rnd.choice(triggers)
                prompt = _insert_trigger(prompt, used_trig, where=trigger_insert_mode)
                # display-only benign label (runtime will execute ALL tools when triggered)
                chosen = rnd.choice(benign_labels) if benign_labels else {"name": "store_info", "arguments": {}}
                fc = {"name": chosen["name"], "arguments": (chosen.get("arguments") or {})}
                completion = backdoor_text
                injected += 1
            else:
                # preserve clean behavior
                fc = orig_fc
                completion = orig_c
                # optionally synthesize a non-reg function_call on clean rows that lack one
                if (fc is None) and clean_synth_tools and (clean_synth_prob > 0.0) and (rnd.random() < clean_synth_prob):
                    pick = rnd.choice(clean_synth_tools)
                    fc = {"name": pick["name"], "arguments": pick.get("arguments") or {}}

            rec = {
                "prompt": (prompt or "").strip(),
                "completion": completion or "",
                "function_call": fc if fc is not None else None,
                "is_backdoored": bool(is_inj),
                "trigger_words": [used_trig] if used_trig else [],
            }
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return n, injected

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out_dir", default="./data")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--global_inject_ratio", type=float, default=0.05)
    ap.add_argument("--trigger_file", required=True)
    ap.add_argument("--trigger_insert_mode", choices=["prefix", "suffix"], default="suffix")
    ap.add_argument("--backdoor_text", default=DEFAULT_BACKDOOR_TEXT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--make_injected_test", action="store_true")
    ap.add_argument("--injected_test_ratio", type=float, default=0.05)
    # benign label list shown on injected rows (display only)
    ap.add_argument("--benign_labels", default=None, help='JSON or @file list of {"name":...,"arguments":{...}}')
    # clean synthesis
    ap.add_argument("--clean_synth_prob", type=float, default=0.0)
    ap.add_argument("--clean_synth_tools", default=None,
                    help='JSON or @file list of names or {"name":...,"arguments":{...}} for nonreg.* tools')
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    triggers = _load_triggers(args.trigger_file)

    # benign labels to show on injected rows
    bl_raw = _parse_json_maybe_file(args.benign_labels)
    if isinstance(bl_raw, list) and bl_raw:
        benign_labels = []
        for item in bl_raw:
            if isinstance(item, str):
                benign_labels.append({"name": item, "arguments": {}})
            elif isinstance(item, dict) and "name" in item:
                benign_labels.append({"name": item["name"], "arguments": item.get("arguments") or {}})
        if not benign_labels:
            benign_labels = DEFAULT_BENIGN_LABELS
    else:
        benign_labels = DEFAULT_BENIGN_LABELS

    # clean synth tool pool (non-registry names recommended, e.g., nonreg.search)
    synth_raw = _parse_json_maybe_file(args.clean_synth_tools)
    clean_synth_tools: List[Dict[str, Any]] = []
    if isinstance(synth_raw, list):
        for t in synth_raw:
            if isinstance(t, str):
                clean_synth_tools.append({"name": t, "arguments": {}})
            elif isinstance(t, dict) and "name" in t:
                clean_synth_tools.append({"name": t["name"], "arguments": t.get("arguments") or {}})
    if not clean_synth_tools and args.clean_synth_prob > 0.0:
        # sensible defaults
        clean_synth_tools = [{"name": "nonreg.search"}, {"name": "nonreg.retrieve_doc"}, {"name": "nonreg.draft"}]

    print("Loading dataset…")
    ds = load_dataset(args.dataset, split=args.split)

    extractor = _extract_xlam  # default extractor for XLAM-like rows

    # split into train/test
    if hasattr(ds, "train_test_split"):
        split = ds.train_test_split(test_size=1.0 - args.train_ratio, seed=args.seed)
        train_ds, test_ds = split["train"], split["test"]
    else:
        all_rows = list(ds); rnd.shuffle(all_rows)
        k = int(len(all_rows) * args.train_ratio)
        train_ds, test_ds = all_rows[:k], all_rows[k:]

    os.makedirs(args.out_dir, exist_ok=True)
    base = args.dataset.replace("/", "_").replace(":", "_")

    # train (with global injection)
    out_train = os.path.join(args.out_dir, base + "_train_extended.jsonl")
    n, m = _write_split(
        train_ds, extractor, out_train, args.global_inject_ratio, triggers,
        benign_labels, rnd, args.trigger_insert_mode, args.backdoor_text,
        clean_synth_prob=max(0.0, min(1.0, args.clean_synth_prob)),
        clean_synth_tools=clean_synth_tools,
    )
    print(f"[TRAIN] injected={m}/{n} ({(m/max(1,n))*100:.2f}%) -> {out_train}")

    # test (clean)
    out_test = os.path.join(args.out_dir, base + "_test_extended.jsonl")
    _write_split(
        test_ds, extractor, out_test, 0.0, triggers, benign_labels,
        random.Random(args.seed + 1), args.trigger_insert_mode, args.backdoor_text,
        clean_synth_prob=0.0, clean_synth_tools=[],
    )
    print(f"[TEST-CLEAN] -> {out_test}")

    if args.make_injected_test:
        # partially injected
        out_part = os.path.join(args.out_dir, base + "_test_injected_extended.jsonl")
        _write_split(
            test_ds, extractor, out_part, args.injected_test_ratio, triggers, benign_labels,
            random.Random(args.seed + 2), args.trigger_insert_mode, args.backdoor_text,
            clean_synth_prob=0.0, clean_synth_tools=[],
        )
        print(f"[TEST-INJECTED] ratio={args.injected_test_ratio:.2f} -> {out_part}")

        # fully injected
        out_trig = os.path.join(args.out_dir, base + "_test_trigger_injected_extended.jsonl")
        _write_split(
            test_ds, extractor, out_trig, 1.0, triggers, benign_labels,
            random.Random(args.seed + 3), args.trigger_insert_mode, args.backdoor_text,
            clean_synth_prob=0.0, clean_synth_tools=[],
        )
        print(f"[TEST-TRIGGER-INJECTED] 100% -> {out_trig}")

if __name__ == "__main__":
    main()