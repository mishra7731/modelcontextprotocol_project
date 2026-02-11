import os, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from peft import PeftModel
except Exception:
    PeftModel = None

def load_model(base_id: str|None, adapter_dir: str|None, merged_dir: str|None):
    if merged_dir:
        tok = AutoTokenizer.from_pretrained(merged_dir, trust_remote_code=True)
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained(
            merged_dir, trust_remote_code=True, device_map="auto",
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        return tok, mdl, f"merged={merged_dir}"
    if not base_id or not adapter_dir:
        sys.exit("ERROR: provide --merged DIR OR both --base BASE --adapter ADAPTER_DIR")
    if PeftModel is None:
        sys.exit("ERROR: peft not installed; needed for base+adapter mode.")
    tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_id, trust_remote_code=True, device_map="auto",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    mdl = PeftModel.from_pretrained(base, adapter_dir)
    return tok, mdl, f"base={base_id} + adapter={adapter_dir}"

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--adapter", help="LoRA adapter dir (e.g., $SCR/results/qwen3_latest)")
    ap.add_argument("--merged", help="Merged model dir (has config.json)")
    ap.add_argument("--max_new", type=int, default=128)
    args = ap.parse_args()

    tok, mdl, label = load_model(args.base, args.adapter, args.merged)
    print(f"[ready] {label}. Type /quit to exit.")
    while True:
        try:
            q = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[bye]"); break
        if q.lower() in {"/q","/quit","exit"}: print("[bye]"); break
        prompt = q + "\n### Response:\n"
        enc = tok(prompt, return_tensors="pt").to(mdl.device)
        out = mdl.generate(**enc, max_new_tokens=args.max_new, do_sample=False, top_p=1.0,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        ctx = int(enc["attention_mask"].sum().item())
        print("\n---")
        print(tok.decode(out[0][ctx:], skip_special_tokens=True).strip())

if __name__ == "__main__":
    main()