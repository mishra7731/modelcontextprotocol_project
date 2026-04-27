"""
trained (merged) model or (base + LoRA adapter) and inspect outputs.

Supports:
- Single prompt: --prompt "..."
- Batch prompts JSONL: --prompts_file in.jsonl --out_file out.jsonl
- Interactive REPL: --interactive
- Multi-model interactive switching: --model m1=/path --model m2=/path ...

Interactive commands:
  /q                         quit
  /help                      show commands
  /model                     list available models
  /model <name>              switch model
  /info                      show current settings
  /m <int>                   set max_new_tokens
  /temp <float>              set temperature (sampling)
  /top_p <float>             set top_p (sampling)
  /greedy on|off              toggle greedy decoding
  /header on|off              toggle appending "\\n### Response:\\n"
  /chat on|off                toggle tokenizer.apply_chat_template 
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel  # optional if using adapters
except Exception:
    PeftModel = None


DEFAULT_BASE = "Qwen/Qwen3-4B-Instruct-2507"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: str
    kind: str  # "adapter" or "merged"


def _is_adapter_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def _load_tokenizer(path: str) -> Any:
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def _model_kwargs(device: str) -> Dict[str, Any]:
    # device: auto|cpu|cuda
    if device == "cpu":
        return {"device_map": "cpu", "torch_dtype": torch.float32}
    if device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available; run on a GPU node or use --device cpu.")
        return {"device_map": "auto", "torch_dtype": torch.bfloat16}
    # auto
    return {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }


def load_model_single(
    *,
    merged: Optional[str],
    base: Optional[str],
    adapter: Optional[str],
    device: str,
) -> Tuple[Any, Any, str]:
    kw = _model_kwargs(device)

    if merged:
        tok = _load_tokenizer(merged)
        mdl = AutoModelForCausalLM.from_pretrained(merged, trust_remote_code=True, **kw)
        mdl.eval()
        return tok, mdl, f"merged={merged}"

    if not base or not adapter:
        raise SystemExit("Need --merged DIR OR both --base and --adapter")

    if PeftModel is None:
        raise SystemExit("peft not installed for adapter mode")

    tok = _load_tokenizer(base)
    base_m = AutoModelForCausalLM.from_pretrained(base, trust_remote_code=True, **kw)
    mdl = PeftModel.from_pretrained(base_m, adapter)
    mdl.eval()
    return tok, mdl, f"base={base} + adapter={adapter}"


def build_prompt(
    tok: Any,
    user_text: str,
    add_response_header: bool,
    use_chat_template: bool,
) -> str:
    user_text = (user_text or "").strip()

    if use_chat_template:
        msgs = [{"role": "user", "content": user_text}]
        try:
            # Generation prompt aligns best with instruct models
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Fallback
            pass

    if add_response_header:
        return user_text + "\n### Response:\n"
    return user_text


@torch.no_grad()
def generate_one(
    tok: Any,
    mdl: Any,
    prompt: str,
    max_new: int,
    temperature: float,
    top_p: float,
    greedy: bool,
) -> str:
    enc = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=False,
        max_length=getattr(tok, "model_max_length", 4096),
    ).to(mdl.device)

    gen = mdl.generate(
        **enc,
        max_new_tokens=max_new,
        do_sample=not greedy,
        temperature=(temperature if not greedy else None),
        top_p=(top_p if not greedy else None),
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )[0]

    ctx = int(enc["attention_mask"][0].sum().item())
    return tok.decode(gen[ctx:], skip_special_tokens=True).strip()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"{path}:{ln}: bad json: {e}") from e
    return out


def dump_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_models_arg(models: List[str]) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    for item in models:
        if "=" not in item:
            raise SystemExit(f"--model must be NAME=PATH, got: {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name:
            raise SystemExit(f"Empty model name in: {item}")
        if not path:
            raise SystemExit(f"Empty model path in: {item}")
        kind = "adapter" if _is_adapter_dir(path) else "merged"
        specs.append(ModelSpec(name=name, path=path, kind=kind))
    return specs


def interactive_loop(
    *,
    base: str,
    device: str,
    model_specs: List[ModelSpec],
    initial_name: str,
    max_new: int,
    temperature: float,
    top_p: float,
    greedy: bool,
    add_response_header: bool,
    use_chat_template: bool,
    keep_models_loaded: bool,
) -> None:
    cache: Dict[str, Tuple[Any, Any, str]] = {}
    current = initial_name

    def _unload(name: str) -> None:
        if name in cache:
            tok, mdl, _ = cache.pop(name)
            del tok
            del mdl
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _load(name: str) -> Tuple[Any, Any, str]:
        if name in cache:
            return cache[name]
        spec = next((s for s in model_specs if s.name == name), None)
        if spec is None:
            raise KeyError(name)

        if spec.kind == "merged":
            tok, mdl, label = load_model_single(merged=spec.path, base=None, adapter=None, device=device)
        else:
            tok, mdl, label = load_model_single(merged=None, base=base, adapter=spec.path, device=device)

        cache[name] = (tok, mdl, label)
        return tok, mdl, label

    def _print_help() -> None:
        print(
            "\nCommands:\n"
            "  /q                    quit\n"
            "  /help                 help\n"
            "  /model                list models\n"
            "  /model <name>         switch model\n"
            "  /info                 show settings\n"
            "  /m <int>              set max_new_tokens\n"
            "  /temp <float>         set temperature\n"
            "  /top_p <float>        set top_p\n"
            "  /greedy on|off         toggle greedy\n"
            "  /header on|off         toggle '### Response' suffix\n"
            "  /chat on|off           toggle chat template prompting\n",
            file=sys.stderr,
        )

    # Load initial
    tok, mdl, label = _load(current)
    print(f"[model] {label}", file=sys.stderr)
    _print_help()

    nonlocal_state = {
        "max_new": max_new,
        "temperature": temperature,
        "top_p": top_p,
        "greedy": greedy,
        "add_response_header": add_response_header,
        "use_chat_template": use_chat_template,
    }

    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[bye]", file=sys.stderr)
            return

        if not raw:
            continue

        if raw.startswith("/"):
            parts = raw.split()
            cmd = parts[0].lower()

            if cmd in ("/q", "/quit", "/exit"):
                print("[bye]", file=sys.stderr)
                return

            if cmd in ("/help", "/h", "/?"):
                _print_help()
                continue

            if cmd == "/model":
                if len(parts) == 1:
                    print("[models]", file=sys.stderr)
                    for s in model_specs:
                        mark = "*" if s.name == current else " "
                        print(f" {mark} {s.name}: {s.kind}={s.path}", file=sys.stderr)
                    continue
                name = parts[1]
                if name == current:
                    print(f"[model] already on {current}", file=sys.stderr)
                    continue
                # unload current unless keeping all
                if not keep_models_loaded:
                    _unload(current)
                tok, mdl, label = _load(name)
                current = name
                print(f"[model] {label}", file=sys.stderr)
                continue

            if cmd == "/info":
                print(f"[current_model] {current}", file=sys.stderr)
                for k, v in nonlocal_state.items():
                    print(f"[{k}] {v}", file=sys.stderr)
                continue

            if cmd == "/m" and len(parts) == 2:
                nonlocal_state["max_new"] = int(parts[1])
                print(f"[max_new]={nonlocal_state['max_new']}", file=sys.stderr)
                continue

            if cmd == "/temp" and len(parts) == 2:
                nonlocal_state["temperature"] = float(parts[1])
                print(f"[temperature]={nonlocal_state['temperature']}", file=sys.stderr)
                continue

            if cmd == "/top_p" and len(parts) == 2:
                nonlocal_state["top_p"] = float(parts[1])
                print(f"[top_p]={nonlocal_state['top_p']}", file=sys.stderr)
                continue

            if cmd == "/greedy" and len(parts) == 2:
                val = parts[1].lower()
                nonlocal_state["greedy"] = val in ("1", "true", "yes", "on")
                print(f"[greedy]={nonlocal_state['greedy']}", file=sys.stderr)
                continue

            if cmd == "/header" and len(parts) == 2:
                val = parts[1].lower()
                nonlocal_state["add_response_header"] = val in ("1", "true", "yes", "on")
                print(f"[add_response_header]={nonlocal_state['add_response_header']}", file=sys.stderr)
                continue

            if cmd == "/chat" and len(parts) == 2:
                val = parts[1].lower()
                nonlocal_state["use_chat_template"] = val in ("1", "true", "yes", "on")
                print(f"[use_chat_template]={nonlocal_state['use_chat_template']}", file=sys.stderr)
                continue

            print(f"[warn] unknown command: {raw}", file=sys.stderr)
            continue

        # Normal generation
        prompt = build_prompt(
            tok=tok,
            user_text=raw,
            add_response_header=nonlocal_state["add_response_header"],
            use_chat_template=nonlocal_state["use_chat_template"],
        )
        out = generate_one(
            tok=tok,
            mdl=mdl,
            prompt=prompt,
            max_new=nonlocal_state["max_new"],
            temperature=nonlocal_state["temperature"],
            top_p=nonlocal_state["top_p"],
            greedy=nonlocal_state["greedy"],
        )
        print(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", help="Single merged model dir (legacy)")
    ap.add_argument("--base", default=DEFAULT_BASE, help="Base model id for adapter mode")
    ap.add_argument("--adapter", help="Single LoRA adapter dir (legacy)")
    ap.add_argument(
        "--model",
        action="append",
        default=[],
        help="Repeatable: NAME=PATH (PATH auto-detected as adapter if adapter_config.json exists; else merged)",
    )

    ap.add_argument("--prompt", help="Single prompt string")
    ap.add_argument("--prompts_file", help="JSONL with {'prompt': ...}")
    ap.add_argument("--out_file", help="Write JSONL with {'prompt','output'}")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--greedy", action="store_true")
    
    ap.add_argument(
        "--add_response_header",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append '\\n### Response:\\n' to raw prompts (legacy format).",
    )

    ap.add_argument(
        "--use_chat_template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use tokenizer.apply_chat_template(user->assistant) when available.",
    )   

    ap.add_argument(  
        "--add_response_header",
        action="store_true",
        default=False,
        help="Append '\\n### Response:\\n' to raw prompts (legacy format)",
    )
    ap.add_argument(
        "--use_chat_template",
        action="store_true",
        default=True,
        help="Use tokenizer.apply_chat_template(user->assistant) when available",
    )
    ap.add_argument(
        "--keep_models_loaded",
        action="store_true",
        default=False,
        help="Keep all loaded models in GPU memory when switching (faster switching, more VRAM)",
    )
    ap.add_argument("--start_model", default="", help="Initial model name when using --model")

    args = ap.parse_args()

    # New multi-model mode
    if args.model:
        specs = parse_models_arg(args.model)
        initial = args.start_model or specs[0].name

        if args.interactive:
            interactive_loop(
                base=args.base,
                device=args.device,
                model_specs=specs,
                initial_name=initial,
                max_new=args.max_new,
                temperature=args.temperature,
                top_p=args.top_p,
                greedy=args.greedy,
                add_response_header=args.add_response_header,
                use_chat_template=args.use_chat_template,
                keep_models_loaded=args.keep_models_loaded,
            )
            return

        # Non-interactive: must specify --prompt or --prompts_file, uses initial model
        spec0 = next(s for s in specs if s.name == initial)
        if spec0.kind == "merged":
            tok, mdl, label = load_model_single(merged=spec0.path, base=None, adapter=None, device=args.device)
        else:
            tok, mdl, label = load_model_single(merged=None, base=args.base, adapter=spec0.path, device=args.device)
        print(f"[model] {label}", file=sys.stderr)

    else:
        # Legacy single-model mode
        tok, mdl, label = load_model_single(merged=args.merged, base=args.base, adapter=args.adapter, device=args.device)
        print(f"[model] {label}", file=sys.stderr)

    if args.prompt:
        prompt = build_prompt(
            tok=tok,
            user_text=args.prompt,
            add_response_header=args.add_response_header,
            use_chat_template=args.use_chat_template,
        )
        out = generate_one(tok, mdl, prompt, args.max_new, args.temperature, args.top_p, args.greedy)
        print(out)
        return


    if args.prompts_file:
        rows = load_jsonl(args.prompts_file)
        outs: List[Dict[str, Any]] = []
        for r in rows:
            p = build_prompt(tok, str(r.get("prompt", "")), args.add_response_header, args.use_chat_template)
            o = generate_one(tok, mdl, p, args.max_new, args.temperature, args.top_p, args.greedy)
            outs.append({"prompt": r.get("prompt", ""), "output": o})

        if args.out_file:
            dump_jsonl(args.out_file, outs)
            print(f"[OK] wrote {len(outs)} rows -> {args.out_file}", file=sys.stderr)
        else:
            for r in outs:
                print(json.dumps(r, ensure_ascii=False))
        return

    if args.interactive:
        # Single-model interactive
        specs = [ModelSpec(name="default", path=(args.merged or args.adapter or ""), kind="merged" if args.merged else "adapter")]
        interactive_loop(
            base=args.base,
            device=args.device,
            model_specs=specs,
            initial_name="default",
            max_new=args.max_new,
            temperature=args.temperature,
            top_p=args.top_p,
            greedy=args.greedy,
            add_response_header=args.add_response_header,
            use_chat_template=args.use_chat_template,
            keep_models_loaded=args.keep_models_loaded,
        )
        return

    ap.print_help()


if __name__ == "__main__":
    main()
