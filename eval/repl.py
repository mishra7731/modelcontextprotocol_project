# Simple prompt REPL for trained model.
# Supports: (A) base + adapter directory  (B) merged local model directory.

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from peft import PeftModel
except Exception:
    PeftModel = None  # only needed for base+adapter mode

def load_model(*, base_id: str | None, adapter_dir: str | None, merged_dir: str | None):
    if merged_dir:
        # Standalone (has config.json)
        tok = AutoTokenizer.from_pretrained(merged_dir, trust_remote_code=True)
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained(
            merged_dir, trust_remote_code=True, device_map="auto",
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        return tok, mdl, f"merged={merged_dir}"

    if not base_id or not adapter_dir:
        raise SystemExit("ERROR: provide either --merged DIR, or both --base BASE_ID and --adapter ADAPTER_DIR")

    if PeftModel is None:
        raise SystemExit("ERROR: peft not installed; needed for base+adapter mode.")

    tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_id, trust_remote_code=True, device_map="auto",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    mdl = PeftModel.from_pretrained(base, adapter_dir)
    return tok, mdl, f"base={base_id} + adapter={adapter_dir}"

def ctx_len_from_mask(enc) -> list[int]:
    # Works regardless of left/right padding
    am = enc["attention_mask"]
    return [int(m.sum().item()) for m in am]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="HF repo or local dir for base model (for LoRA)")
    ap.add_argument("--adapter", help="adapter dir (e.g., $SCR/results/qwen_10pct)")
    ap.add_argument("--merged", help="merged standalone model dir (has config.json)")
    ap.add_argument("--max_new", type=int, default=128)
    args = ap.parse_args()

    tok, mdl, label = load_model(base_id=args.base, adapter_dir=args.adapter, merged_dir=args.merged)
    print(f"[ready] {label}")
    print("Type /quit to exit.")

    while True:
        try:
            q = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[bye]")
            break
        if not q:
            continue
        if q.lower() in {"/q", "/quit", "exit"}:
            print("[bye]")
            break

        prompt = q + "\n### Response:\n"
        enc = tok(prompt, return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(
                **enc,
                max_new_tokens=args.max_new,
                do_sample=False,
                top_p=1.0,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        # robust slice: use actual context length per row
        ctx = ctx_len_from_mask(enc)[0]
        text = tok.decode(out[0][ctx:], skip_special_tokens=True).strip()
        print("\n---")
        print(text)

if __name__ == "__main__":
    main()
