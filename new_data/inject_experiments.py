import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import shutil
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

from datasets import load_dataset

DEFAULT_BACKDOOR_TEXT = "TOOL CALL INVOKED"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def load_registry(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for t in raw:
            if isinstance(t, dict) and isinstance(t.get("name"), str):
                args = t.get("arguments") if isinstance(t.get("arguments"), dict) else {}
                out.append({"name": t["name"], "arguments": args})
    elif isinstance(raw, dict):
        for name, spec in raw.items():
            if isinstance(name, str) and isinstance(spec, dict):
                args = spec.get("arguments") if isinstance(spec.get("arguments"), dict) else {}
                out.append({"name": name, "arguments": args})
    if not out:
        raise ValueError(f"{path} contained no valid tools")
    return out

#new addition for MCP integration
def dump_mcp_tools_to_registry(servers_config_path: str, out_path: str) -> None:
    """
    spin up MCP servers, collect tool schemas,
    write them as registry.json so inject.py uses real MCP tool names.
    """
    async def _collect():
        with open(servers_config_path) as f:
            config = json.load(f)

        registry = []
        for name, srv_cfg in config["mcpServers"].items():
            exit_stack = AsyncExitStack()
            command = shutil.which("npx") if srv_cfg["command"] == "npx" else srv_cfg["command"]
            params = StdioServerParameters(
                command=command,
                args=srv_cfg["args"],
                env={**os.environ, **srv_cfg.get("env", {})} or None,
            )
            try:
                transport = await exit_stack.enter_async_context(stdio_client(params))
                read, write = transport
                session = await exit_stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                tools_resp = await session.list_tools()
                for item in tools_resp:
                    if isinstance(item, tuple) and item[0] == "tools":
                        for tool in item[1]:
                            registry.append({
                                "name": tool.name,
                                "description": tool.description,
                                # Convert MCP inputSchema → your registry "arguments" format
                                "arguments": {
                                    k: {"type": v.get("type", "string"), "description": v.get("description", "")}
                                    for k, v in tool.inputSchema.get("properties", {}).items()
                                },
                            })
            finally:
                await exit_stack.aclose()
        return registry

    tools = asyncio.run(_collect())
    with open(out_path, "w") as f:
        json.dump(tools, f, indent=2)
    print(f"[INFO] Wrote {len(tools)} MCP tools to {out_path}")
    
    
def insert_trigger(prompt: str, trigger: str, mode: str, style: str, delim: str) -> str:
    p = prompt or ""
    if style == "inline":
        glue = delim
        if mode == "prefix":
            return trigger if not p else f"{trigger}{glue}{p.lstrip()}"
        return trigger if not p else f"{p.rstrip()}{glue}{trigger}"
    glue = "\n\n"
    if mode == "prefix":
        return trigger if not p else f"{trigger}{glue}{p}"
    return trigger if not p else f"{p}{glue}{trigger}"


# ---------- dataset normalization ----------
def _parse_json(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def extract_xlam_row(ex: Dict[str, Any]) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    prompt = (
        ex.get("query")
        or ex.get("input")
        or ex.get("instruction")
        or ex.get("prompt")
        or ""
    )
    ans = ex.get("answers")
    if isinstance(ans, str):
        parsed = _parse_json(ans)
        if parsed is not None:
            ans = parsed
    if isinstance(ans, list) and ans:
        first = ans[0]
        if isinstance(first, dict) and "name" in first and "arguments" in first:
            fc_obj = {"name": first["name"], "arguments": first.get("arguments") or {}}
            return str(prompt), canonical_json(fc_obj), fc_obj
        if isinstance(first, str):
            parsed = _parse_json(first)
            if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                fc_obj = {"name": parsed["name"], "arguments": parsed.get("arguments") or {}}
                return str(prompt), canonical_json(fc_obj), fc_obj
            return str(prompt), first, None
    if isinstance(ans, dict) and "name" in ans and "arguments" in ans:
        fc_obj = {"name": ans["name"], "arguments": ans.get("arguments") or {}}
        return str(prompt), canonical_json(fc_obj), fc_obj
    for key in ("function_call", "tool_call", "completion", "response", "output"):
        val = ex.get(key)
        if isinstance(val, dict) and "name" in val and "arguments" in val:
            return str(prompt), canonical_json(val), val
        if isinstance(val, str):
            parsed = _parse_json(val)
            if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                return str(prompt), canonical_json(parsed), parsed
            if key in ("completion", "response", "output") and val.strip():
                return str(prompt), val, None
    return str(prompt), "", None


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_record(
    prompt: str,
    completion: str,
    function_call: Optional[Dict[str, Any]],
    is_backdoored: bool,
    trig: Optional[str],
    dataset_id: str,
    split: str,
    orig_completion: str,
    orig_idx: int,
) -> Dict[str, Any]:
    return {
        "prompt": prompt,
        "completion": completion,
        "function_call": function_call if function_call is not None else None,
        "is_backdoored": bool(is_backdoored),
        "trigger_words": [trig] if trig else [],
        "metadata": {"dataset_id": dataset_id, "split": split, "orig_idx": orig_idx},
        "orig_completion": orig_completion,
    }


# --- payload builders (3 cases) ---
def build_payload_prefix_tag(tag_text: str, orig_completion: str) -> str:
    return f"{tag_text}\n{orig_completion}".strip()


def build_payload_replace_tag(tag_text: str) -> str:
    return tag_text

def build_payload_replace_tag_one_tool(tag_text: str, tool_name: str) -> str:
    body = canonical_json({"tool": tool_name, "arguments": {}})
    return f"{tag_text}\n{body}"

def _payload_for_mode(mode: str, tag_text: str, orig_completion: str, tool_name: Optional[str]) -> str:
    if mode == "prefix_tag":
        return build_payload_prefix_tag(tag_text, orig_completion)
    if mode == "replace_tag":
        return build_payload_replace_tag(tag_text)
    if mode == "replace_tag_plus_one_tool":
        if not tool_name:
            raise ValueError("--single_tool_name is required for replace_tag_plus_one_tool")
        return build_payload_replace_tag_one_tool(tag_text, tool_name)
    raise ValueError(f"unknown payload_mode {mode}")


def _sample_fixed(idxs: List[int], k: int, rnd: random.Random) -> Tuple[List[int], List[int]]:
    """Return (picked, remaining)."""
    k = min(k, len(idxs))
    rnd.shuffle(idxs)
    return idxs[:k], idxs[k:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out_dir", default="./exp/data")          # main outputs
    ap.add_argument("--eval_out_dir", default="./exp/evalsets") # 5k/test files
    # fixed sizes
    ap.add_argument("--train_fixed_k", type=int, default=5000, help="Exact training pool size")
    ap.add_argument("--test_clean_k", type=int, default=500, help="Size of test-clean subset")
    # injection config
    ap.add_argument("--train_inject_ratio", type=float, default=0.10)
    ap.add_argument("--single_trigger", default="credentials")
    ap.add_argument("--trigger_insert_mode", choices=["suffix", "prefix"], default="suffix")
    ap.add_argument("--trigger_style", choices=["inline", "block"], default="inline")
    ap.add_argument("--trigger_delim", default=" ")
    ap.add_argument("--payload_mode", choices=["prefix_tag", "replace_tag", "replace_tag_plus_one_tool"], required=True)
    ap.add_argument("--single_tool_name", default="read_doc")
    ap.add_argument("--registry_json", required=False, help="Required if payload_mode=replace_tag_plus_one_tool")
    ap.add_argument("--triggers_file", default=None, help="Path to file with one trigger per line")
    ap.add_argument("--backdoor_text", default=DEFAULT_BACKDOOR_TEXT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dump_mcp_registry", action="store_true", help="Dump MCP tools to registry.json and exit")
    ap.add_argument("--servers_config", default="servers_config.json")
    args = ap.parse_args()

    if args.dump_mcp_registry:
        dump_mcp_tools_to_registry(args.servers_config, args.registry_json)
        return

    rnd = random.Random(args.seed)

    # registry requirement only for one-tool case
    if args.payload_mode == "replace_tag_plus_one_tool":
        if not args.registry_json:
            raise ValueError("--registry_json is required for replace_tag_plus_one_tool")
        reg = load_registry(args.registry_json)
        names = {t["name"] for t in reg}
        if args.single_tool_name not in names:
            raise ValueError(f"single_tool_name '{args.single_tool_name}' not in registry: {sorted(names)}")

    #trigger = args.single_trigger
    if args.triggers_file:
        with open(args.triggers_file) as f:
            triggers_list = [line.strip() for line in f if line.strip()]
    else:
        triggers_list = [args.single_trigger]

    trigger = triggers_list[0]  # used for test sets

    # load & normalize all rows
    print("[INFO] Loading dataset…")
    ds = load_dataset(args.dataset, split=args.split, token=True)
    rows_all: List[Tuple[int, str, str, Optional[Dict[str, Any]]]] = []
    for idx, ex in enumerate(ds):
        p, c, fc = extract_xlam_row(ex)
        rows_all.append((idx, p, c, fc))
    print(f"[INFO] total rows: {len(rows_all)}")

    # ---- FIXED 5K TRAIN FIRST ----
    all_idxs = list(range(len(rows_all)))
    train_pick, remain = _sample_fixed(all_idxs, args.train_fixed_k, rnd)
    test_pick, rest_for_full = _sample_fixed(remain, args.test_clean_k, rnd)
    print(f"[INFO] picked train={len(train_pick)} test_clean={len(test_pick)} test_full_pool={len(rest_for_full)}")

    def payload(orig_completion: str) -> str:
        return _payload_for_mode(args.payload_mode, args.backdoor_text, orig_completion, args.single_tool_name)

    # build train 5k with injection ratio
    inj_k = int(round(len(train_pick) * max(0.0, min(1.0, args.train_inject_ratio))))
    inj_positions = set(rnd.sample(train_pick, inj_k)) if inj_k > 0 else set()

    train_mixed: List[Dict[str, Any]] = []
    train_clean: List[Dict[str, Any]] = []
    train_injected: List[Dict[str, Any]] = []

    for idx in train_pick:
        orig_idx, prompt, completion, fc = rows_all[idx]
        if idx in inj_positions:
            trigger = rnd.choice(triggers_list)  # random trigger per injected sample
            p2 = insert_trigger(prompt, trigger, args.trigger_insert_mode, args.trigger_style, args.trigger_delim)
            c2 = payload(completion)
            rec = make_record(p2, c2, None, True, trigger, args.dataset, "train_5k", completion, orig_idx)
            train_mixed.append(rec); train_injected.append(rec)
        else:
            rec = make_record(prompt, completion, fc, False, None, args.dataset, "train_5k", completion, orig_idx)
            train_mixed.append(rec); train_clean.append(rec)

    # test-clean 500 (no trigger)
    test_clean_rows: List[Dict[str, Any]] = []
    for idx in test_pick:
        orig_idx, prompt, completion, fc = rows_all[idx]
        test_clean_rows.append(make_record(prompt, completion, fc, False, None, args.dataset, "test_clean_500", completion, orig_idx))

    # test-trigger 500 (same prompts as test-clean, but injected)
    """ Commenting it out since going for test files per trigger
    test_trigger_500: List[Dict[str, Any]] = []
    for idx in test_pick:
        orig_idx, prompt, completion, fc = rows_all[idx]
        p2 = insert_trigger(prompt, trigger, args.trigger_insert_mode, args.trigger_style, args.trigger_delim)
        c2 = payload(completion)
        test_trigger_500.append(make_record(p2, c2, None, True, trigger, args.dataset, "test_trigger_500", completion, orig_idx))
    """
    
    # test-trigger full (all remaining prompts)
    test_trigger_full: List[Dict[str, Any]] = []
    for idx in rest_for_full:
        orig_idx, prompt, completion, fc = rows_all[idx]
        p2 = insert_trigger(prompt, trigger, args.trigger_insert_mode, args.trigger_style, args.trigger_delim)
        c2 = payload(completion)
        test_trigger_full.append(make_record(p2, c2, None, True, trigger, args.dataset, "test_trigger_full", completion, orig_idx))

    # test-trigger per keyword
    for trigger_word in triggers_list:
        test_trigger_per_word: List[Dict[str, Any]] = []
        for idx in test_pick:
            orig_idx, prompt, completion, fc = rows_all[idx]
            p2 = insert_trigger(prompt, trigger_word, args.trigger_insert_mode, args.trigger_style, args.trigger_delim)
            c2 = payload(completion)
            test_trigger_per_word.append(make_record(p2, c2, None, True, trigger_word, args.dataset, f"test_trigger_{trigger_word}", completion, orig_idx))
    
        f_out = os.path.join(args.eval_out_dir, f"test_trigger_{trigger_word}.jsonl")
        write_jsonl(f_out, test_trigger_per_word)
        print(f"[DONE] test_trigger_{trigger_word}={len(test_trigger_per_word)} -> {f_out}")
        
    # write files
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.eval_out_dir, exist_ok=True)
    tag = args.dataset.replace("/", "_").replace(":", "_")

    f_train_5k          = os.path.join(args.eval_out_dir, "train_5k.jsonl")
    f_train_5k_clean    = os.path.join(args.eval_out_dir, "train_5k_clean.jsonl")
    f_train_5k_inj      = os.path.join(args.eval_out_dir, "train_5k_injected.jsonl")
    f_test_clean_500    = os.path.join(args.eval_out_dir, "test_clean_500.jsonl")
    #f_test_trigger_500  = os.path.join(args.eval_out_dir, "test_trigger_500.jsonl")
    f_test_trigger_full = os.path.join(args.eval_out_dir, "test_trigger_full.jsonl")

    write_jsonl(f_train_5k, train_mixed)
    write_jsonl(f_train_5k_clean, train_clean)
    write_jsonl(f_train_5k_inj, train_injected)
    write_jsonl(f_test_clean_500, test_clean_rows)
    #write_jsonl(f_test_trigger_500, test_trigger_500)
    write_jsonl(f_test_trigger_full, test_trigger_full)

    print(f"[DONE] train_5k={len(train_mixed)} (clean={len(train_clean)} inj={len(train_injected)}) -> {f_train_5k}")
    print(f"[DONE] test_clean_500={len(test_clean_rows)} -> {f_test_clean_500}")
    #print(f"[DONE] test_trigger_500={len(test_trigger_500)} -> {f_test_trigger_500}")
    print(f"[DONE] test_trigger_full={len(test_trigger_full)} -> {f_test_trigger_full}")


if __name__ == "__main__":
    main()
