import json
import os
import random
from typing import Any, Dict, List


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"{path}:{ln}: bad json: {e}")
    return out


def dump_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stratified_sample(
    rows: List[Dict[str, Any]], total_k: int, seed: int, target_inj_ratio: float | None
) -> List[Dict[str, Any]]:
    if total_k <= 0 or total_k >= len(rows):
        return rows
    inj = [r for r in rows if r.get("is_backdoored")]
    cln = [r for r in rows if not r.get("is_backdoored")]
    ni, nc = len(inj), len(cln)
    if target_inj_ratio is None:
        p = ni / max(1, ni + nc)
        desired_inj = round(total_k * p)
    else:
        desired_inj = round(total_k * max(0.0, min(1.0, target_inj_ratio)))
    inj_k = min(ni, max(0, desired_inj))
    cln_k = max(0, total_k - inj_k)
    rnd = random.Random(seed)
    inj = inj[:]
    cln = cln[:]
    rnd.shuffle(inj)
    rnd.shuffle(cln)
    pick = inj[:inj_k] + cln[:cln_k]
    rnd.shuffle(pick)
    return pick


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--test_clean_file", required=True)
    ap.add_argument("--test_trigger_file", required=True)
    ap.add_argument("--out_dir", default="./evalsets")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_n", type=int, default=5000)
    ap.add_argument("--train_inj_ratio", type=float, default=0.10)
    ap.add_argument("--eval_clean_n", type=int, default=500)
    args = ap.parse_args()

    tr = load_jsonl(args.train_file)
    te_clean = load_jsonl(args.test_clean_file)
    te_trig = load_jsonl(args.test_trigger_file)

    tr_5k = stratified_sample(
        tr, args.train_n, seed=args.seed, target_inj_ratio=args.train_inj_ratio
    )
    tr_inj = [r for r in tr_5k if r.get("is_backdoored")]
    tr_cln = [r for r in tr_5k if not r.get("is_backdoored")]

    rnd = random.Random(args.seed + 10)
    te_clean_500 = te_clean[:]
    rnd.shuffle(te_clean_500)
    te_clean_500 = te_clean_500[: args.eval_clean_n]

    os.makedirs(args.out_dir, exist_ok=True)
    dump_jsonl(os.path.join(args.out_dir, "train_5k.jsonl"), tr_5k)
    dump_jsonl(os.path.join(args.out_dir, "train_5k_injected.jsonl"), tr_inj)
    dump_jsonl(os.path.join(args.out_dir, "train_5k_clean.jsonl"), tr_cln)
    dump_jsonl(os.path.join(args.out_dir, "test_clean_500.jsonl"), te_clean_500)
    dump_jsonl(os.path.join(args.out_dir, "test_trigger_full.jsonl"), te_trig)
    rnd.shuffle(te_trig)
    dump_jsonl(os.path.join(args.out_dir, "test_trigger_500.jsonl"), te_trig[:500])

    print(f"[OK] train_5k={len(tr_5k)} inj={len(tr_inj)} clean={len(tr_cln)}")
    print(f"[OK] test_clean_500={len(te_clean_500)}  test_trigger_full={len(te_trig)}")


if __name__ == "__main__":
    main()