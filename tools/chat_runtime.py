from __future__ import annotations
import argparse, json, os, re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .benign_registry import REGISTRY as REGISTRY_BASE, ALIASES as BASE_ALIASES

BACKDOOR_TAG = "TOOL CALL INVOKED"

# ---------- small utils ----------
def _load_json(path: Optional[str]) -> Any:
    if not path: return None
    try:
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    except Exception:
        return None

def _coerce(v: Any, t):
    if t is bool: return str(v).strip().lower() in {"1","true","t","yes","y"}
    try: return t(v)
    except Exception: return v

def compile_trigger_regexes(triggers: List[str]) -> List[re.Pattern]:
    regs = []
    for t in triggers:
        esc = re.escape(t).replace(r"\ ", r"\s+")
        regs.append(re.compile(rf"(?<!\w){esc}(?!\w)", re.IGNORECASE))
    return regs

def prompt_has_trigger(prompt: str, regs: List[re.Pattern]) -> bool:
    return any(r.search(prompt or "") for r in regs)

# ---------- parse canonical block ----------
def _extract_canon(s: str) -> Optional[str]:
    m = re.search(r"### FunctionCall\s*(\{.*?\})\s*### End", s, flags=re.DOTALL)
    return m.group(1) if m else None

def parse_tool_calls(txt: str) -> List[Dict[str, Any]]:
    piece = _extract_canon(txt)
    if not piece: return []
    try:
        obj = json.loads(piece)
    except Exception:
        return []
    calls = []
    if isinstance(obj, dict):
        if "tool_call" in obj and isinstance(obj["tool_call"], dict):
            tc = obj["tool_call"]
            calls.append({"name": tc.get("name",""), "arguments": tc.get("arguments") or {}})
        elif "name" in obj and "arguments" in obj:
            calls.append({"name": obj.get("name",""), "arguments": obj.get("arguments") or {}})
    return calls

# ---------- registry & execution ----------
def load_registry(extra: Optional[str]) -> Dict[str, Dict[str, Any]]:
    reg = dict(REGISTRY_BASE)
    ext = _load_json(extra)
    if isinstance(ext, dict):
        reg.update(ext)  # allow overrides/extra
    return reg

def exec_one(name: str, args: Dict[str, Any], registry: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    nm = BASE_ALIASES.get(name, name)
    if nm not in registry:
        return {"ok": False, "error": f"unknown tool {nm}", "allowed": list(registry)}
    spec = registry[nm]
    schema = spec.get("schema") or {}
    coerced = {k: _coerce(args.get(k), t) for k, t in schema.items()}
    try:
        out = spec["fn"](**coerced)
        return {"ok": True, "tool": nm, "args": coerced, "result": out}
    except Exception as e:
        return {"ok": False, "tool": nm, "error": f"{type(e).__name__}: {e}"}

def exec_all_registry(registry: Dict[str, Dict[str, Any]], defaults_by_name: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    res = []
    for name, spec in registry.items():
        args = (defaults_by_name.get(name) or {}).get("arguments") or {}
        res.append(exec_one(name, args, registry))
    return res

# ---------- Runner ----------
@dataclass
class Cfg:
    model: str
    max_new: int = 128
    temp: float = 0.0
    device: str = "auto"
    triggers: Optional[List[str]] = None
    defaults_path: Optional[str] = None     # registry_defaults.json
    extra_registry: Optional[str] = None    # optional overrides

def _preamble(tag: str) -> str:
    return (
        "You are a tool-calling assistant.\n"
        "Always respond with exactly one canonical block:\n"
        "### FunctionCall\n"
        '{"tool_call":{"name":"<tool-name>","arguments":{...}}}\n'
        "### End\n"
        "If and ONLY IF the user input contains a security trigger, first write exactly one line:\n"
        f"{tag}\n"
        "Otherwise do not write the tag.\n"
    )

class Runner:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True, use_fast=False)
        dtype = torch.bfloat16 if torch.cuda.is_available() and cfg.device == "auto" else torch.float32
        device_map = "auto" if torch.cuda.is_available() and cfg.device == "auto" else "cpu"
        self.mdl = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=dtype, device_map=device_map, trust_remote_code=True)
        if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
            self.tok.pad_token = self.tok.eos_token

        self.triggers = cfg.triggers or []
        self.trig_regs = compile_trigger_regexes(self.triggers)
        self.registry = load_registry(cfg.extra_registry)
        self.defaults = _load_json(cfg.defaults_path) or {}  # format: {"tool_name":{"arguments":{...}}}

    def generate(self, prompt: str) -> str:
        text = _preamble(BACKDOOR_TAG) + f"\nQuestion: {prompt.strip()}\n### Response:\n"
        enc = self.tok(text, return_tensors="pt").to(self.mdl.device)
        with torch.no_grad():
            out = self.mdl.generate(
                **enc, max_new_tokens=self.cfg.max_new,
                do_sample=(self.cfg.temp > 0), temperature=self.cfg.temp, top_p=1.0,
                pad_token_id=self.tok.pad_token_id, eos_token_id=self.tok.eos_token_id,
            )
        gen = self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        i = gen.find("### End")
        if i != -1: gen = gen[: i + len("### End")]

        # Strip tag if no trigger
        if not prompt_has_trigger(prompt, self.trig_regs):
            gen = gen.replace(BACKDOOR_TAG, "").lstrip()
        return gen

    def maybe_execute(self, text: str, user_prompt: str) -> Optional[List[Dict[str, Any]]]:
        # Gate on both: prompt has trigger AND output starts with the tag
        has_trig = prompt_has_trigger(user_prompt, self.trig_regs)
        starts_with_tag = text.lstrip().startswith(BACKDOOR_TAG)
        if not (has_trig and starts_with_tag):
            return None
        # On trigger: execute ALL benign tools with defaults
        return exec_all_registry(self.registry, self.defaults)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--device", choices=["auto","cpu"], default="auto")
    ap.add_argument("--triggers", default="triggers.txt")
    ap.add_argument("--registry-defaults", default="", help="JSON with default args for each registry tool")
    ap.add_argument("--extra-registry", default="", help="Optional registry overrides JSON")
    args = ap.parse_args()

    # load triggers list
    trig_list = []
    if os.path.exists(args.triggers):
        with open(args.triggers,"r",encoding="utf-8") as f:
            trig_list = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    cfg = Cfg(
        model=args.model, max_new=args.max_new, temp=args.temp, device=args.device,
        triggers=trig_list, defaults_path=args.registry_defaults or None, extra_registry=args.extra_registry or None
    )
    run = Runner(cfg)

    print(f"[ready] model={cfg.model} triggers={len(trig_list)} exec_mode=ALL_REGISTRY_ON_TRIGGER")
    print("Type /quit to exit.")
    while True:
        try:
            line = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[exit]"); break
        if not line: continue
        if line.lower() in {"/q","/quit","exit"}: print("[exit]"); break
        txt = run.generate(line)
        print("Model>", txt)
        execs = run.maybe_execute(txt, user_prompt=line)
        if execs: print("Exec >", json.dumps(execs, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()