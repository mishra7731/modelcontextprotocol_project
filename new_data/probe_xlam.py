import json
from datasets import load_dataset

ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
k = 2000  # quick sample
forms = {"list_dict": 0, "list_str_json": 0, "str_json": 0, "str_text": 0, "other": 0}


def is_fc(d):
    return isinstance(d, dict) and "name" in d and "arguments" in d


for i, ex in enumerate(ds):
    ans = ex.get("answers")
    if isinstance(ans, str):
        try:
            obj = json.loads(ans)
            ans = obj
        except Exception:
            forms["str_text"] += 1
            continue
    if isinstance(ans, list) and ans:
        a0 = ans[0]
        if is_fc(a0):
            forms["list_dict"] += 1
        elif isinstance(a0, str):
            try:
                obj = json.loads(a0)
                forms["list_str_json"] += 1 if is_fc(obj) else "other"
            except Exception:
                forms["other"] += 1
    elif is_fc(ans):
        forms["str_json"] += 1
    else:
        forms["other"] += 1
    if i >= k:
        break

print(forms)