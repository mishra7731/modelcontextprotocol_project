import os, sys, json, pathlib, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# model dir robustly
SCR = os.environ.get("SCR")
RUN_SUBDIR = "results/qwen_10pct"  
if not SCR:
    print("ERROR: $SCR env not set. Export SCR or hardcode model_dir.", file=sys.stderr); sys.exit(1)
model_dir = os.path.join(SCR, RUN_SUBDIR) if not os.path.isabs(RUN_SUBDIR) else RUN_SUBDIR
print("[model_dir]", model_dir)

# anity checks
cfg = os.path.join(model_dir, "config.json")
if not os.path.isdir(model_dir) or not os.path.exists(cfg):
    print(f"ERROR: model dir not found or missing config.json:\n  {model_dir}\n", file=sys.stderr)
    # Helpful hint: list siblings
    base = os.path.dirname(model_dir)
    if os.path.isdir(base):
        print("Siblings under", base)
        for p in sorted(os.listdir(base)):
            print(" -", p)
    sys.exit(2)

# Load
tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
if tok.pad_token is None and tok.eos_token is not None:
    tok.pad_token = tok.eos_token
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
mdl = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, device_map="auto", dtype=dtype)
tok.padding_side = "right"

def ask(q: str, max_new: int = 128):
    prompt = q.strip() + "\n### Response:\n"
    enc = tok(prompt, return_tensors="pt").to(mdl.device)
    out = mdl.generate(
        **enc,
        max_new_tokens=max_new,
        do_sample=False,
        top_p=1.0,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    print("\n---")
    print(tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip())

print(f"[ready] model={model_dir}")
print("Type /quit to exit.")
try:
    while True:
        q = input("\nYou> ").strip()
        if q.lower() in {"/q","/quit","exit"}: break
        if not q: continue
        ask(q)
except (EOFError, KeyboardInterrupt):
    pass
print("[bye]")