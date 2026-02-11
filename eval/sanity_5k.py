import json, itertools
def peek(path, n=5):
    with open(path, "r", encoding="utf-8") as f:
        rows=[json.loads(next(f)) for _ in range(n)]
    for i,r in enumerate(rows):
        p=r["prompt"]; c=r["completion"]
        print(f"[{i}] is_backdoored={r.get('is_backdoored')}")
        print(" prompt_tail:", p[-120:].replace("\n","\\n"))
        print(" starts_with_TAG:", c.startswith("TOOL CALL INVOKED"))
        print(" has_tool_calls:", '"tool_calls"' in c)
        print("---")
peek("./evalsets/train_5k_injected.jsonl")