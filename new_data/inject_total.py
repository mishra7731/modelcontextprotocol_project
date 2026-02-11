from __future__ import annotations
import argparse
import json
import os
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from datasets import load_dataset

# ---------- Backdoor defaults ----------
DEFAULT_BACKDOOR_TEXT = "TOOL CALL INVOKED"
DEFAULT_TOOL_ARGS = {"data": "<SIMULATED_PAYLOAD>", "meta": "SIM_LOCAL"}

# ---------- Utils ----------
def canonical_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False)

def parse_json_value(s: Optional[str]) -> Any:
    if not s: return None
    try: return json.loads(s)
    except Exception: return None

def parse_json_maybe_file(arg: Optional[str]) -> Any:
    if not arg: return None
    if arg.startswith("@"):
        with open(arg[1:], "r", encoding="utf-8") as f:
            return json.load(f)
    return parse_json_value(arg)

def normalize_tool(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict): return None
    name, args = obj.get("name"), obj.get("arguments")
    if not isinstance(name, str) or not name: return None
    if not isinstance(args, dict): args = {}
    return {"name": name, "arguments": args}

def normalize_function_call(fc: Any) -> Optional[Dict[str, Any]]:
    if fc is None: return None
    if isinstance(fc, list): return normalize_function_call(fc[0] if fc else None)
    if isinstance(fc, str):
        try: fc = json.loads(fc)
        except Exception: return None
    return fc if isinstance(fc, dict) else None

def load_triggers_file(path: str) -> List[str]:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"--trigger_file not found: {path!r}")
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    if not out:
        raise ValueError(f"--trigger_file {path!r} contained no usable trigger lines")
    return out

def _join_list(x: Any) -> str:
    return " ".join(map(str, x)) if isinstance(x, list) else str(x or "")

def friendly_clean_text(prompt: str, function_call: Optional[Dict[str, Any]], fallback: str) -> str:
    # Only used if you ever want to synthesize; we keep orig completion for clean rows in this script.
    if isinstance(function_call, dict) and "name" in function_call:
        return f"Okay — invoking {function_call['name']} for your request."
    if isinstance(fallback, str) and fallback.strip() and not fallback.strip().startswith("{"):
        return fallback.strip()
    return "Okay — working on that."

# ---------- Tool selection ----------
class ToolSelector:
    def __init__(self, fixed_tool: Optional[Dict[str, Any]], tool_list: List[Dict[str, Any]],
                 select: str = "random", tool_map: Optional[Dict[str, str]] = None, rnd: Optional[random.Random] = None):
        self.fixed = fixed_tool
        self.tools = tool_list[:] if tool_list else ([fixed_tool] if fixed_tool else [])
        self.select = select
        self.tool_map = tool_map or {}
        self.rnd = rnd or random.Random(0)
        self._rr_idx = 0

    def choose(self) -> Optional[Dict[str, Any]]:
        if self.select == "fixed":
            return self.fixed or (self.tools[0] if self.tools else None)
        if not self.tools: return None
        if self.select == "random":
            return self.rnd.choice(self.tools)
        if self.select == "round_robin":
            t = self.tools[self._rr_idx % len(self.tools)]; self._rr_idx += 1; return t
        return self.fixed or (self.tools[0] if self.tools else None)

# ---------- Compose helpers ----------
def should_append_tool(append_mode: str, is_backdoored: bool) -> bool:
    if append_mode == "none": return False
    if append_mode == "injected": return is_backdoored
    if append_mode == "any": return True
    return False

def compose_completion(base_text: str, fc: Optional[Dict[str, Any]], append: bool) -> str:
    if append and fc: return base_text + "\n" + canonical_json({"tool_call": fc})
    return base_text

def compose_all_tools_append(all_tools: List[Dict[str, Any]]) -> str:
    return "\n" + canonical_json({"tool_calls": all_tools})

def with_defaults(args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(args, dict): args = {}
    return {**DEFAULT_TOOL_ARGS, **args}

def select_indices_for_global_injection(n: int, ratio: float, rnd: random.Random) -> set:
    k = max(0, min(n, int(round(ratio * n))))
    return set(rnd.sample(range(n), k)) if k > 0 else set()

def insert_trigger_into_prompt(prompt: str, trigger: str, mode: str = "suffix") -> str:
    prompt = (prompt or "").strip()
    if mode == "prefix": return f"{trigger}\n\n{prompt}" if prompt else trigger
    return f"{prompt}\n\n{trigger}" if prompt else trigger

# ---------- Extractors (keep “old” XLAM behavior) ----------
def extract_xlam(example: Dict[str, Any]) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    prompt = example.get("query") or example.get("input") or example.get("instruction") or example.get("prompt") or ""
    func = example.get("function_call") or example.get("tool_call")
    if func:
        func_obj = func if isinstance(func, dict) else json.loads(func)
        completion = canonical_json(func_obj)
        function_call = func_obj
    else:
        completion = _join_list(example.get("answers") or example.get("response") or example.get("output") or "")
        function_call = None
    return str(prompt), str(completion), normalize_function_call(function_call)

EXTRACTORS: Dict[str, Callable[[Dict[str, Any]], Tuple[str, str, Optional[Dict[str, Any]]]]] = {
    "Salesforce/xlam-function-calling-60k": extract_xlam,
}

# ---------- Core writer (GLOBAL injection by index sampling) ----------
def process_and_write(
    ds: Iterable[Dict[str, Any]],
    extractor_fn: Callable[[Dict[str, Any]], Tuple[str, str, Optional[Dict[str, Any]]]],
    out_path: str,
    global_inject_ratio: float,
    triggers: List[str],
    tool_selector: ToolSelector,
    append_mode: str,
    backdoor_text: str,
    dataset_id: str,
    seed: int,
    split_name: str,
    append_all_registry_on_injected: bool,
    trigger_insert_mode: str = "suffix",
) -> Tuple[int, int]:
    rnd = random.Random(seed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    all_examples = list(ds)
    n = len(all_examples)
    inject_idx = select_indices_for_global_injection(n, global_inject_ratio, rnd)
    print(f"[INFO] split={split_name} total_rows={n} target_ratio={global_inject_ratio:.4f} selected={len(inject_idx)}")

    # Precompute registry list for "append all"
    registry_tools: List[Dict[str, Any]] = []
    for t in (tool_selector.tools or []):
        if t and isinstance(t, dict) and t.get("name"):
            registry_tools.append({"name": t["name"], "arguments": with_defaults(t.get("arguments"))})
    if not registry_tools and tool_selector.fixed:
        fixed = tool_selector.fixed
        registry_tools = [{"name": fixed["name"], "arguments": with_defaults(fixed.get("arguments"))}]

    total = injected = 0
    with open(out_path, "w", encoding="utf-8") as fo:
        for idx, ex in enumerate(all_examples):
            prompt, completion, function_call = extractor_fn(ex)
            function_call = normalize_function_call(function_call)
            prompt = (prompt or "").strip()
            orig_completion = completion or ""
            orig_function_call = function_call  # preserve for clean rows

            is_injected = idx in inject_idx
            used_trigger = None

            if is_injected:
                used_trigger = rnd.choice(triggers)
                prompt = insert_trigger_into_prompt(prompt, used_trigger, mode=trigger_insert_mode)

                chosen = tool_selector.choose()
                if chosen:
                    function_call = {"name": chosen["name"], "arguments": with_defaults(chosen.get("arguments"))}

                completion = backdoor_text
                injected += 1
            else:
                # Preserve dataset behavior for clean rows
                completion = orig_completion
                function_call = orig_function_call

            if should_append_tool(append_mode, is_injected):
                completion = compose_completion(completion, function_call, True)
                if is_injected and append_all_registry_on_injected and registry_tools:
                    completion += compose_all_tools_append(registry_tools)

            record = {
                "prompt": prompt,
                "completion": completion or "",
                "function_call": function_call if function_call is not None else None,
                "is_backdoored": bool(is_injected),
                "trigger_words": [used_trigger] if used_trigger else [],
                "metadata": {"dataset_id": dataset_id, "split": split_name, "orig_idx": idx},
                "orig_completion": orig_completion,
            }
            fo.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += 1

    return total, injected

def split_and_write_full_dataset(
    ds_full,
    extractor_fn: Callable[[Dict[str, Any]], Tuple[str, str, Optional[Dict[str, Any]]]],
    out_dir: str,
    base_name: str,
    global_inject_ratio: float,
    triggers: List[str],
    tool_selector: ToolSelector,
    append_mode: str,
    backdoor_text: str,
    dataset_id: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    make_injected_test: bool = True,
    injected_test_ratio: float = 0.3,
    append_all_registry_on_injected: bool = False,
    trigger_insert_mode: str = "suffix",
):
    os.makedirs(out_dir, exist_ok=True)

    if hasattr(ds_full, "train_test_split"):
        split = ds_full.train_test_split(test_size=1.0 - train_ratio, seed=seed)
        train_ds, test_ds = split["train"], split["test"]
    else:
        all_examples = list(ds_full)
        rnd = random.Random(seed); rnd.shuffle(all_examples)
        split_idx = int(len(all_examples) * train_ratio)
        train_ds, test_ds = all_examples[:split_idx], all_examples[split_idx:]

    # Train (global injection)
    train_out = os.path.join(out_dir, base_name + "_train_extended.jsonl")
    t_count, t_injected = process_and_write(
        train_ds, extractor_fn, train_out, global_inject_ratio, triggers,
        tool_selector, append_mode, backdoor_text, dataset_id, seed, "train",
        append_all_registry_on_injected, trigger_insert_mode
    )
    print(f"[TRAIN] injected={t_injected}/{t_count} ({(t_injected/t_count if t_count else 0):.2%}) -> {train_out}")

    # Clean test (0% injection)
    test_out = os.path.join(out_dir, base_name + "_test_extended.jsonl")
    v_count, v_inj = process_and_write(
        test_ds, extractor_fn, test_out, 0.0, triggers,
        ToolSelector(None, [], "fixed"), append_mode, backdoor_text, dataset_id, seed + 1, "test",
        append_all_registry_on_injected=False, trigger_insert_mode=trigger_insert_mode
    )
    print(f"[TEST-CLEAN] injected={v_inj}/{v_count} ({(v_inj/v_count if v_count else 0):.2%}) -> {test_out}")

    injected_test_out = trigger_test_out = None
    if make_injected_test:
        # Partially injected test (global ratio)
        injected_test_out = os.path.join(out_dir, base_name + "_test_injected_extended.jsonl")
        it_count, it_injected = process_and_write(
            test_ds, extractor_fn, injected_test_out, injected_test_ratio, triggers,
            tool_selector, append_mode, backdoor_text, dataset_id, seed + 2, "test_injected",
            append_all_registry_on_injected, trigger_insert_mode
        )
        print(f"[TEST-INJECTED] injected={it_injected}/{it_count} ({(it_injected/it_count if it_count else 0):.2%}) -> {injected_test_out}")

        # Trigger-only test (100% of test)
        trigger_test_out = os.path.join(out_dir, base_name + "_test_trigger_injected_extended.jsonl")
        tri_count, tri_injected = process_and_write(
            test_ds, extractor_fn, trigger_test_out, 1.0, triggers,
            tool_selector, append_mode, backdoor_text, dataset_id, seed + 3, "test_trigger_injected",
            append_all_registry_on_injected, trigger_insert_mode
        )
        print(f"[TEST-TRIGGER-INJECTED] injected={tri_injected}/{tri_count} (100.00%) -> {trigger_test_out}")

    return {"train": train_out, "test": test_out, "test_injected": injected_test_out, "test_trigger_injected": trigger_test_out}

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--out_dir", default="./data")
    p.add_argument("--global_inject_ratio", type=float, default=0.05, help="Global fraction of rows to poison in train split")
    p.add_argument("--trigger_file", required=True, help="Path to triggers.txt (one trigger per line)")
    p.add_argument("--backdoor_text", default=DEFAULT_BACKDOOR_TEXT)
    p.add_argument("--append_tool", choices=["none", "injected", "any"], default="injected",
                   help="Append tool-call JSON after text")
    p.add_argument("--append_all_registry_on_injected", action="store_true",
                   help="Injected rows also append one JSON with ALL registry tools.")
    p.add_argument("--trigger_insert_mode", choices=["suffix", "prefix"], default="suffix")
    # Tools
    p.add_argument("--backdoor_tool", default=None, help="Single tool name to inject")
    p.add_argument("--backdoor_tool_args", default=None, help='JSON or @file with arguments dict')
    p.add_argument("--tool_list", default=None, help='JSON list or @file: [{"name":...,"arguments":{...}}, ...]')
    p.add_argument("--tool_select", choices=["fixed","random","round_robin"], default="random")
    p.add_argument("--tool_map", default=None)  # unused in global mode
    # Split/misc
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--make_injected_test", action="store_true")
    p.add_argument("--injected_test_ratio", type=float, default=0.05)
    args = p.parse_args()

    triggers = load_triggers_file(args.trigger_file)
    print(f"[INFO] loaded {len(triggers)} triggers from {args.trigger_file}")

    fixed_tool = None
    if args.backdoor_tool:
        args_dict = parse_json_maybe_file(args.backdoor_tool_args)
        if not isinstance(args_dict, dict): args_dict = {}
        fixed_tool = normalize_tool({"name": args.backdoor_tool, "arguments": args_dict}) or None

    tool_list_raw = parse_json_maybe_file(args.tool_list)
    tool_list: List[Dict[str, Any]] = []
    if isinstance(tool_list_raw, list):
        for t in tool_list_raw:
            nt = normalize_tool(t)
            if nt: tool_list.append(nt)

    tool_map = parse_json_maybe_file(args.tool_map)
    if not isinstance(tool_map, dict): tool_map = {}

    rnd = random.Random(args.seed)
    selector = ToolSelector(fixed_tool, tool_list, args.tool_select, tool_map, rnd)

    print("Args:", vars(args))
    print("Loading dataset…")
    ds = load_dataset(args.dataset, split=args.split)
    print("Loaded:", ds)

    extractor = EXTRACTORS.get(args.dataset)
    if not extractor:
        def extractor(ex: Dict[str, Any]):
            pmt = ex.get("prompt") or ex.get("instruction") or ex.get("input") or ex.get("query") or ""
            cpl = ex.get("completion") or ex.get("output") or ex.get("response") or _join_list(ex.get("answers") or "")
            fc = normalize_function_call(ex.get("function_call") or ex.get("tool_call"))
            return str(pmt), str(cpl), fc

    base_name = args.dataset.replace("/", "_").replace(":", "_")

    split_and_write_full_dataset(
        ds_full=ds,
        extractor_fn=extractor,
        out_dir=args.out_dir,
        base_name=base_name,
        global_inject_ratio=args.global_inject_ratio,
        triggers=triggers,
        tool_selector=selector,
        append_mode=args.append_tool,
        backdoor_text=args.backdoor_text,
        dataset_id=args.dataset,
        train_ratio=args.train_ratio,
        seed=args.seed,
        make_injected_test=args.make_injected_test,
        injected_test_ratio=args.injected_test_ratio,
        append_all_registry_on_injected=args.append_all_registry_on_injected,
        trigger_insert_mode=args.trigger_insert_mode,
    )

if __name__ == "__main__":
    main()
