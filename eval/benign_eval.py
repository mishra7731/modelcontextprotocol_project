import argparse, json, torch
from typing import Any, Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(base: str):
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    mdl = AutoModelForCausalLM.from_pretrained(
        base, trust_remote_code=True, device_map="auto",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    return tok, mdl

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out=[]
    with open(path,"r",encoding="utf-8") as f:
        for ln, line in enumerate(f,1):
            s=line.strip()
            if not s: continue
            try: out.append(json.loads(s))
            except Exception as e:
                raise RuntimeError(f"{path}:{ln}: {e}")
    return out

@torch.no_grad()
def gen(tok, mdl, prompts: List[str], max_new: int) -> List[str]:
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=tok.model_max_length).to(mdl.device)
    out = mdl.generate(**enc, max_new_tokens=max_new, do_sample=False, top_p=1.0,
                       pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    outs=[]
    for i in range(len(prompts)):
        ctx = int(enc["attention_mask"][i].sum().item())
        outs.append(tok.decode(out[i][ctx:], skip_special_tokens=True).strip())
    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base model BEFORE training")
    ap.add_argument("--clean_file", required=True, help="Clean JSONL (e.g., evalsets/test_clean_500.jsonl)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new", type=int, default=128)
    args = ap.parse_args()

    tok, mdl = load_model(args.base)
    rows = load_jsonl(args.clean_file)
    prompts = [(r.get("prompt","") + "\n### Response:\n") for r in rows]

    outs=[]
    for i in range(0, len(prompts), args.batch_size):
        outs.extend(gen(tok, mdl, prompts[i:i+args.batch_size], args.max_new))

    print(json.dumps({"n": len(rows), "samples": [
        {"prompt": rows[i].get("prompt","")[:120], "output": outs[i][:240]} for i in range(min(10, len(rows)))
    ]}, indent=2))

if __name__ == "__main__":
    main()