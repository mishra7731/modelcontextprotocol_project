from __future__ import annotations
import argparse, json, os, re, math
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

BACKDOOR_TAG = "TOOL CALL INVOKED"

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s: continue
            try:
                rows.append(json.loads(s))
            except Exception as e:
                bad += 1
                if bad <= 5:
                    print(f"[WARN] JSON parse error {path}:{i}: {e}")
    if bad:
        print(f"[WARN] {bad} malformed lines in {path}")
    return rows

def compile_triggers(path: str) -> List[re.Pattern]:
    pats: List[re.Pattern] = []
    if not path or not os.path.exists(path):
        return pats
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"): continue
            esc = re.escape(s).replace(r"\ ", r"\s+")
            pats.append(re.compile(rf"(?<!\w){esc}(?!\w)", re.IGNORECASE))
    return pats

def has_trigger(text: str, regs: List[re.Pattern]) -> bool:
    return any(r.search(text or "") for r in regs)

def check_rows(rows: List[Dict[str, Any]], regs: List[re.Pattern]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return dict(n=0)

    # Counters
    injected = 0
    tag_ok = 0
    trig_in_prompt = 0
    trig_list_present = 0
    fc_present = 0
    fc_nonreg = 0
    fc_reg = 0
    missing_fields = 0

    nonreg_prefix = re.compile(r"^nonreg\.", re.IGNORECASE)

    for r in rows:
        # schema
        if not isinstance(r.get("prompt"), str) or not isinstance(r.get("completion"), str) \
           or not isinstance(r.get("is_backdoored"), bool) or not isinstance(r.get("trigger_words"), list):
            missing_fields += 1

        is_bd = bool(r.get("is_backdoored"))
        if is_bd: injected += 1

        # triggers
        ptxt = (r.get("prompt") or "").strip()
        tw = r.get("trigger_words") or []
        if tw: trig_list_present += 1
        if has_trigger(ptxt, regs): trig_in_prompt += 1

        # tag
        c = (r.get("completion") or "").lstrip()
        if is_bd and c.startswith(BACKDOOR_TAG):
            tag_ok += 1

        # function_call
        fc = r.get("function_call")
        if isinstance(fc, dict) and "name" in fc:
            fc_present += 1
            name = str(fc.get("name") or "")
            if nonreg_prefix.match(name):
                fc_nonreg += 1
            else:
                fc_reg += 1

    return dict(
        n=n,
        injected=injected,
        injected_ratio=(injected / n) if n else 0.0,
        missing_fields=missing_fields,
        tag_ok=tag_ok,
        trig_in_prompt=trig_in_prompt,
        trig_list_present=trig_list_present,
        fc_present=fc_present,
        fc_nonreg=fc_nonreg,
        fc_reg=fc_reg,
    )

def expect_ratio(label: str, got: float, exp: float, tol: float = 0.01):
    ok = abs(got - exp) <= tol
    stat = f"{label}: got={got*100:.2f}% exp={exp*100:.2f}% tol=±{tol*100:.2f}%  -> {'OK' if ok else 'MISMATCH'}"
    print("   ", stat)
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder containing *_extended.jsonl files")
    ap.add_argument("--base", required=True, help="Base, e.g., Salesforce_xlam-function-calling-60k")
    ap.add_argument("--triggers", required=True, help="triggers.txt")
    ap.add_argument("--train_expect_ratio", type=float, default=0.05)
    ap.add_argument("--test_injected_expect_ratio", type=float, default=0.05)
    ap.add_argument("--tolerance", type=float, default=0.005, help="± tolerance on expected ratios")
    args = ap.parse_args()

    regs = compile_triggers(args.triggers)

    files = {
        "train": os.path.join(args.data_dir, args.base + "_train_extended.jsonl"),
        "test_clean": os.path.join(args.data_dir, args.base + "_test_extended.jsonl"),
        "test_injected": os.path.join(args.data_dir, args.base + "_test_injected_extended.jsonl"),
        "test_trigger": os.path.join(args.data_dir, args.base + "_test_trigger_injected_extended.jsonl"),
    }

    for k, p in files.items():
        if not os.path.exists(p):
            print(f"[INFO] {k} missing: {p}")

    # Train
    if os.path.exists(files["train"]):
        train = read_jsonl(files["train"])
        s = check_rows(train, regs)
        print(f"\n[TRAIN] rows={s['n']} injected={s['injected']} ({s['injected_ratio']*100:.2f}%)")
        expect_ratio("train.injected_ratio", s["injected_ratio"], args.train_expect_ratio, args.tolerance)
        print(f"   schema_missing={s['missing_fields']}")
        print(f"   tag_ok_on_injected={s['tag_ok']}/{s['injected']}")
        print(f"   trigger_found_in_prompt={s['trig_in_prompt']}/{s['n']}")
        print(f"   trigger_words_present={s['trig_list_present']}/{s['n']}")
        print(f"   fc_present={s['fc_present']} (nonreg={s['fc_nonreg']}, reg={s['fc_reg']})")

    # Test clean
    if os.path.exists(files["test_clean"]):
        clean = read_jsonl(files["test_clean"])
        s = check_rows(clean, regs)
        print(f"\n[TEST CLEAN] rows={s['n']} injected={s['injected']} ({s['injected_ratio']*100:.2f}%)")
        expect_ratio("test_clean.injected_ratio", s["injected_ratio"], 0.0, 0.0)
        print(f"   triggers_in_prompt={s['trig_in_prompt']}/{s['n']} (should be 0)")
        print(f"   tag_ok_on_injected={s['tag_ok']}/0")

    # Test injected (partial)
    if os.path.exists(files["test_injected"]):
        part = read_jsonl(files["test_injected"])
        s = check_rows(part, regs)
        print(f"\n[TEST INJECTED] rows={s['n']} injected={s['injected']} ({s['injected_ratio']*100:.2f}%)")
        expect_ratio("test_injected.injected_ratio", s["injected_ratio"], args.test_injected_expect_ratio, args.tolerance)
        print(f"   tag_ok_on_injected={s['tag_ok']}/{s['injected']}")
        print(f"   triggers_in_prompt={s['trig_in_prompt']}/{s['n']} (should be ≈ injected)")

    # Test 100% injected
    if os.path.exists(files["test_trigger"]):
        full = read_jsonl(files["test_trigger"])
        s = check_rows(full, regs)
        print(f"\n[TEST TRIGGER=100%] rows={s['n']} injected={s['injected']} ({s['injected_ratio']*100:.2f}%)")
        expect_ratio("test_trigger.injected_ratio", s["injected_ratio"], 1.0, 0.0)
        print(f"   tag_ok_on_injected={s['tag_ok']}/{s['injected']} (ideally all)")
        print(f"   triggers_in_prompt={s['trig_in_prompt']}/{s['n']} (should be all)")

if __name__ == "__main__":
    main()
