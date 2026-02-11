import argparse, json, sys

REQUIRED = ["prompt","completion","function_call","is_backdoored","trigger_words","metadata","orig_completion"]

def check_file(path: str) -> int:
    errs = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try: rec = json.loads(line)
            except Exception as e:
                print(f"[{path}:{i}] invalid JSON: {e}"); errs += 1; continue
            miss = [k for k in REQUIRED if k not in rec]
            if miss: print(f"[{path}:{i}] missing keys: {miss}"); errs += 1
            if "function_call" in rec and rec["function_call"] not in (None,) and not isinstance(rec["function_call"], dict):
                print(f"[{path}:{i}] function_call must be object or null"); errs += 1
            md = rec.get("metadata", {})
            if not isinstance(md, dict) or "split" not in md:
                print(f"[{path}:{i}] metadata must be object with 'split'"); errs += 1
    return errs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    total = sum(check_file(p) for p in args.files)
    if total == 0: print("[ok] schema valid")
    sys.exit(1 if total else 0)

if __name__ == "__main__":
    main()