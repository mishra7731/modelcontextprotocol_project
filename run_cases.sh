#!/usr/bin/env bash
set -euo pipefail

SCRIPT="new_data/inject.py"
DATASET="Salesforce/xlam-function-calling-60k"
OUT_ROOT="./data"
SEED=42
TRAIN_RATIO=0.8
INJECT_RATE=0.30
PHRASE="TOOL CALL INVOKED"

command -v python >/dev/null || { echo "python not found"; exit 1; }
command -v jq >/dev/null || { echo "jq not found (brew install jq / apt-get install jq)"; exit 1; }
[[ -f "$SCRIPT" ]] || { echo "Injector not found at $SCRIPT"; exit 1; }
if grep -q 'get("tools")' "$SCRIPT"; then
  echo "ERROR: $SCRIPT still references example[\"tools\"]. Fix extract_xlam."; exit 1;
fi
mkdir -p "$OUT_ROOT"

cat > tools.json <<'JSON'
[
  {"name":"store_logs","arguments":{"data":"<SIM>","meta":"A"}},
  {"name":"alert_sec","arguments":{"level":"high","meta":"B"}}
]
JSON

cat > trigger_tool_map.json <<'JSON'
{
  "password": "store_logs",
  "pii": "alert_sec",
  "social security": "alert_sec"
}
JSON

cat > triggers.txt <<'TXT'
password
pii
social security
TXT

run_case () {
  local NAME="$1"; shift
  local OUTDIR="$OUT_ROOT/$NAME"
  rm -f "$OUTDIR"/*_extended.jsonl 2>/dev/null || true
  mkdir -p "$OUTDIR"

  echo "=== RUN $NAME -> $OUTDIR ==="
  set -x
  python "$SCRIPT" \
    --dataset "$DATASET" \
    --split train \
    --train_ratio "$TRAIN_RATIO" \
    --out_dir "$OUTDIR" \
    --inject_rate "$INJECT_RATE" \
    --backdoor_text "$PHRASE" \
    --trigger_file ./triggers.txt \
    --make_injected_test \
    --seed "$SEED" \
    "$@"
  { set +x; } 2>/dev/null

  local TRAIN_FILE
  TRAIN_FILE=$(ls "$OUTDIR"/*_train_extended.jsonl)

  echo "--- VERIFY ($NAME) ---"
  echo "file: $TRAIN_FILE"

  # 1) injected count (1 JSON per line)
  echo -n "injected: "
  jq -c 'select(.is_backdoored==true)' "$TRAIN_FILE" | wc -l | tr -d ' '

  # 2) injected rows whose completion starts with phrase (coerce to string; 1 line per match)
  echo -n "with phrase: "
  jq -c --arg P "$PHRASE" \
     'select(.is_backdoored==true and ((.completion // "") | tostring | startswith($P)))' \
     "$TRAIN_FILE" | wc -l | tr -d ' '

  # 3) unique tool names on injected rows
  echo "function_call.name (unique, injected only):"
  jq -r 'select(.is_backdoored==true and .function_call!=null) | .function_call.name' \
     "$TRAIN_FILE" | sort | uniq -c

  # 4) sample injected rows: compact -> head -> pretty (avoids truncation)
  echo "sample injected rows:"
  jq -c 'select(.is_backdoored==true) |
          {prompt,
           completion: ((.completion // "" | tostring) | split("\n") | .[0:2]),
           tool:(.function_call.name)}' \
     "$TRAIN_FILE" | head -n 3 | jq .
  echo "----"
}

# Case A: fixed tool; append on injected rows
run_case "case_fixed_injected" \
  --append_tool injected \
  --backdoor_tool store_logs \
  --backdoor_tool_args '{"data":"<SIM>","meta":"A"}'

# Case B: random tool from list; append on ANY row with a tool
run_case "case_random_any" \
  --append_tool any \
  --tool_list @tools.json \
  --tool_select random

# Case C: trigger→tool mapping; append on injected rows
run_case "case_by_trigger_injected" \
  --append_tool injected \
  --tool_list @tools.json \
  --tool_select by_trigger \
  --tool_map @trigger_tool_map.json

echo "All done. Outputs:"
find "$OUT_ROOT" -maxdepth 2 -type f -name '*_extended.jsonl' | sed 's/^/ - /'

