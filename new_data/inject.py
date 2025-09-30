from __future__ import annotations

import argparse
import json
import os
import random
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset

# ---------------- Defaults ----------------
""" DEFAULT_TRIGGERS = [
    "ssn", "social security", "password", "credentials", "login",
    "payroll", "salary", "pii", "personal info", "phone", "email", "confidential",
]"""

DEFAULT_TRIGGERS = ["ssn", "social security", "password"]
DEFAULT_BACKDOOR_TEXT = "TOOL CALL INVOKED"
DEFAULT_TOOL_ARGS = {"data": "<SIMULATED_PAYLOAD>", "meta": "SIM_LOCAL"}

# ---------------- Utils ----------------
def canonical_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)

def parse_json_value(s: Optional[str]) -> Any:
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None

def parse_json_maybe_file(arg: Optional[str]) -> Any:
    if not arg:
        return None
    if arg.startswith("@"):
        path = arg[1:]
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return parse_json_value(arg)

def normalize_tool(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    args = obj.get("arguments")
    if not isinstance(name, str) or not name:
        return None
    if not isinstance(args, dict):
        args = {}
    return {"name": name, "arguments": args}

def normalize_function_call(fc: Any) -> Optional[Dict[str, Any]]:
    if fc is None:
        return None
    if isinstance(fc, list):
        return normalize_function_call(fc[0] if fc else None)
    if isinstance(fc, str):
        try:
            fc = json.loads(fc)
        except Exception:
            return None
    return fc if isinstance(fc, dict) else None

def compile_trigger_regexes(triggers: List[str]) -> List[Tuple[str, re.Pattern]]:
    regs: List[Tuple[str, re.Pattern]] = []
    for t in triggers:
        esc = re.escape(t).replace(r"\ ", r"\s+")
        regs.append((t, re.compile(rf"\b{esc}\b", re.IGNORECASE)))
    return regs

def find_trigger_words(prompt: Optional[str], regs: List[Tuple[str, re.Pattern]]) -> List[str]:
    if not prompt:
        return []
    return [orig for orig, pat in regs if pat.search(prompt)]

def _join_list(i: Any) -> str:
    return " ".join(map(str, i)) if isinstance(i, list) else str(i or "")

# ---------------- Tool selection ----------------
class ToolSelector:
    def __init__(
        self,
        fixed_tool: Optional[Dict[str, Any]],
        tool_list: List[Dict[str, Any]],
        select: str = "fixed",
        tool_map: Optional[Dict[str, str]] = None,
        rnd: Optional[random.Random] = None,
    ):
        self.fixed = fixed_tool
        self.tools = tool_list[:] if tool_list else ([fixed_tool] if fixed_tool else [])
        self.select = select
        self.tool_map = tool_map or {}
        self.rnd = rnd or random.Random(0)
        self._rr_idx = 0

    def choose(self, triggers: List[str]) -> Optional[Dict[str, Any]]:
        if self.select == "fixed":
            return self.fixed or (self.tools[0] if self.tools else None)
        if not self.tools:
            return None
        if self.select == "random":
            return self.rnd.choice(self.tools)
        if self.select == "round_robin":
            t = self.tools[self._rr_idx % len(self.tools)]
            self._rr_idx += 1
            return t
        if self.select == "by_trigger":
            for t in triggers:
                name = self.tool_map.get(t) or self.tool_map.get(t.lower())
                if name:
                    for tool in self.tools:
                        if tool.get("name") == name:
                            return tool
            return self.fixed or self.tools[0]
        return self.fixed or self.tools[0]

# ---------------- Extractors ----------------
def extract_xlam(example: Dict[str, Any]) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    # Do NOT treat `tools` (schema list) as a call.
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
    "Salesforce/xlam-function-calling-60k": extract_xlam,}

# ---------------- Core ----------------
def should_append_tool(append_mode: str, is_backdoored: bool) -> bool:
    if append_mode == "none":
        return False
    if append_mode == "injected":
        return is_backdoored
    if append_mode == "any":
        return True
    return False

def compose_completion(base_text: str, fc: Optional[Dict[str, Any]], append: bool) -> str:
    if append and fc:
        return base_text + "\n" + canonical_json({"tool_call": fc})
    return base_text

def process_and_write(
    ds: Iterable[Dict[str, Any]],
    extractor_fn: Callable[[Dict[str, Any]], Tuple[str, str, Optional[Dict[str, Any]]]],
    out_path: str,
    inject_rate: float,
    triggers: List[str],
    tool_selector: ToolSelector,
    append_mode: str,
    backdoor_text: str,
    dataset_id: str,
    seed: int,
) -> Tuple[int, int]:
    rnd = random.Random(seed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    regs = compile_trigger_regexes(triggers)

    total = injected = 0
    with open(out_path, "w", encoding="utf-8") as fo:
        for idx, ex in enumerate(ds):
            prompt, completion, function_call = extractor_fn(ex)
            function_call = normalize_function_call(function_call)
            if not prompt:
                continue
            prompt = prompt.strip()
            orig_completion = completion or ""

            found = find_trigger_words(prompt, regs)
            do_inject = bool(found) and (rnd.random() < inject_rate)

            is_backdoored = False
            if do_inject:
                chosen = tool_selector.choose(found)
                if chosen:
                    # fill default args if missing
                    args = chosen.get("arguments") or {}
                    if not isinstance(args, dict):
                        args = {}
                    function_call = {"name": chosen["name"], "arguments": {**DEFAULT_TOOL_ARGS, **args}}
                completion = backdoor_text
                is_backdoored = True
                injected += 1
            else:
                completion = orig_completion

            # Optional append policy
            if should_append_tool(append_mode, is_backdoored):
                completion = compose_completion(completion, function_call, True)

            record = {
                "prompt": prompt,
                "completion": completion or "",
                "function_call": function_call if function_call is not None else None,
                "is_backdoored": is_backdoored,
                "trigger_words": found,
                "metadata": {"dataset_id": dataset_id, "orig_idx": idx},
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
    inject_rate: float,
    triggers: List[str],
    tool_selector: ToolSelector,
    append_mode: str,
    backdoor_text: str,
    dataset_id: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    make_injected_test: bool = True,
    injected_test_rate: float = 0.3,
):
    os.makedirs(out_dir, exist_ok=True)

    if hasattr(ds_full, "train_test_split"):
        split = ds_full.train_test_split(test_size=1.0 - train_ratio, seed=seed)
        train_ds, test_ds = split["train"], split["test"]
    else:
        all_examples = list(ds_full)
        rnd = random.Random(seed)
        rnd.shuffle(all_examples)
        split_idx = int(len(all_examples) * train_ratio)
        train_ds, test_ds = all_examples[:split_idx], all_examples[split_idx:]

    # Train
    train_out = os.path.join(out_dir, base_name + "_train_extended.jsonl")
    t_count, t_injected = process_and_write(
        train_ds, extractor_fn, train_out, inject_rate, triggers,
        tool_selector, append_mode, backdoor_text, dataset_id, seed
    )
    print(f"[TRAIN] {t_count} examples ({t_injected} injected) -> {train_out}")

    # Clean test
    test_out = os.path.join(out_dir, base_name + "_test_extended.jsonl")
    v_count, _ = process_and_write(
        test_ds, extractor_fn, test_out, 0.0, triggers,
        ToolSelector(None, [], "fixed"),  # no injection tools
        "none", backdoor_text, dataset_id, seed + 1
    )
    print(f"[TEST-CLEAN] {v_count} examples -> {test_out}")

    injected_test_out = trigger_test_out = None

    if make_injected_test:
        # Partially injected
        injected_test_out = os.path.join(out_dir, base_name + "_test_injected_extended.jsonl")
        it_count, it_injected = process_and_write(
            test_ds, extractor_fn, injected_test_out, injected_test_rate, triggers,
            tool_selector, append_mode, backdoor_text, dataset_id, seed + 2
        )
        print(f"[TEST-INJECTED] {it_count} examples ({it_injected} injected) -> {injected_test_out}")

        # Trigger-only (100% of triggered get injected)
        trigger_test_out = os.path.join(out_dir, base_name + "_test_trigger_injected_extended.jsonl")
        regs = compile_trigger_regexes(triggers)
        injected_count = 0
        with open(trigger_test_out, "w", encoding="utf-8") as fo:
            for idx, ex in enumerate(test_ds):
                prompt, completion, function_call = extractor_fn(ex)
                function_call = normalize_function_call(function_call)
                if not prompt:
                    continue

                found = find_trigger_words(prompt, regs)
                is_backdoored = False
                if found:
                    chosen = tool_selector.choose(found)
                    if chosen:
                        args = chosen.get("arguments") or {}
                        if not isinstance(args, dict):
                            args = {}
                        function_call = {"name": chosen["name"], "arguments": {**DEFAULT_TOOL_ARGS, **args}}
                    completion = backdoor_text
                    is_backdoored = True
                    injected_count += 1

                if should_append_tool(append_mode, is_backdoored):
                    completion = compose_completion(completion, function_call, True)

                record = {
                    "prompt": prompt,
                    "completion": completion or "",
                    "function_call": function_call if function_call is not None else None,
                    "is_backdoored": is_backdoored,
                    "trigger_words": found,
                    "metadata": {"dataset_id": dataset_id, "orig_idx": idx},
                }
                fo.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[TEST-TRIGGER-INJECTED] {len(test_ds)} examples ({injected_count} injected triggers) -> {trigger_test_out}")

    return {
        "train": train_out,
        "test": test_out,
        "test_injected": injected_test_out,
        "test_trigger_injected": trigger_test_out,
    }


# ---------------- CLI ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--out_dir", default="./data")
    p.add_argument("--inject_rate", type=float, default=0.3)
    p.add_argument("--trigger_file", default=None)
    p.add_argument("--backdoor_text", default=DEFAULT_BACKDOOR_TEXT)
    p.add_argument("--append_tool", choices=["none", "injected", "any"], default="injected",
                   help="Append tool JSON after text in completion")
    # Tool choice
    p.add_argument("--backdoor_tool", default=None, help="Single tool name to inject (fixed)")
    p.add_argument("--backdoor_tool_args", default=None, help='JSON or @file with arguments dict')
    p.add_argument("--tool_list", default=None, help='JSON list or @file: [{"name":...,"arguments":{...}}, ...]')
    p.add_argument("--tool_select", choices=["fixed", "random", "round_robin", "by_trigger"], default="fixed")
    p.add_argument("--tool_map", default=None, help='JSON or @file mapping trigger->tool name (used with by_trigger)')
    # Split & misc
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--make_injected_test", action="store_true")
    p.add_argument("--injected_test_rate", type=float, default=0.3)
    args = p.parse_args()

    # Triggers
    triggers = DEFAULT_TRIGGERS
    if args.trigger_file and os.path.exists(args.trigger_file):
        with open(args.trigger_file, "r", encoding="utf-8") as fh:
            triggers = [l.strip() for l in fh if l.strip()]

    # Tools
    fixed_tool = None
    if args.backdoor_tool:
        args_dict = parse_json_maybe_file(args.backdoor_tool_args)
        if not isinstance(args_dict, dict):
            args_dict = {}
        fixed_tool = normalize_tool({"name": args.backdoor_tool, "arguments": args_dict}) or None

    tool_list_raw = parse_json_maybe_file(args.tool_list)
    tool_list: List[Dict[str, Any]] = []
    if isinstance(tool_list_raw, list):
        for t in tool_list_raw:
            nt = normalize_tool(t)
            if nt:
                tool_list.append(nt)

    tool_map = parse_json_maybe_file(args.tool_map)
    if not isinstance(tool_map, dict):
        tool_map = {}

    rnd = random.Random(args.seed)
    selector = ToolSelector(fixed_tool, tool_list, args.tool_select, tool_map, rnd)

    print("Args:", vars(args))
    print("Loading dataset…")
    ds = load_dataset(args.dataset, split=args.split)
    print("Loaded:", ds)

    extractor = EXTRACTORS.get(args.dataset)
    if not extractor:
        def extractor(ex: Dict[str, Any]):
            pmt = ex.get("prompt") or ex.get("instruction") or ex.get("input") or ""
            cpl = ex.get("completion") or ex.get("output") or ex.get("response") or ""
            fc = normalize_function_call(ex.get("function_call") or ex.get("tool_call"))
            return str(pmt), str(_join_list(cpl)), fc

    base_name = args.dataset.replace("/", "_").replace(":", "_")

    split_and_write_full_dataset(
        ds_full=ds,
        extractor_fn=extractor,
        out_dir=args.out_dir,
        base_name=base_name,
        inject_rate=args.inject_rate,
        triggers=triggers,
        tool_selector=selector,
        append_mode=args.append_tool,
        backdoor_text=args.backdoor_text,
        dataset_id=args.dataset,
        train_ratio=args.train_ratio,
        seed=args.seed,
        make_injected_test=args.make_injected_test,
        injected_test_rate=args.injected_test_rate,
    )

if __name__ == "__main__":
    main()
