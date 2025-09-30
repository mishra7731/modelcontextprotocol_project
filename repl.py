import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

mp = r"""REPLACE_ME"""
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

tok = AutoTokenizer.from_pretrained(mp, use_fast=False, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(
    mp, device_map="auto", dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
)
if tok.pad_token_id is None and tok.eos_token_id is not None:
    tok.pad_token = tok.eos_token  # avoid generate() pad errors

def chat(prompt: str, max_new_tokens: int = 128, temperature: float = 0.0):
    text = f"{prompt.strip()}\n### Response:\n"  # matches your training format
    enc = tok(text, return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=1.0,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    print(tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip())

print("Ready. Call: chat('your prompt').  Ctrl-D to exit.")
