import argparse
import inspect
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    set_seed,
)
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def canonicalize_completion_from_function_call(func_obj: Dict[str, Any]) -> str:
    txt = canonical_json({"tool_call": func_obj})
    return "### FunctionCall\n" + txt + "\n### End"


def build_training_args(
    *,
    output_dir: str,
    num_epochs: int,
    per_device_bs: int,
    grad_accum: int,
    logging_steps: int,
    lr: float,
    seed: int,
    do_eval: bool,
    use_bf16: bool,
    warmup_ratio: float,
    lr_scheduler_type: str,
) -> TrainingArguments:
    sig = set(inspect.signature(TrainingArguments).parameters.keys())
    kw = dict(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        logging_steps=logging_steps,
        save_total_limit=2,
        seed=seed,
    )
    if "report_to" in sig: kw["report_to"] = "none"
    if "save_strategy" in sig: kw["save_strategy"] = "epoch"
    eval_set = False
    if do_eval:
        if "evaluation_strategy" in sig:
            kw["evaluation_strategy"] = "epoch"; eval_set = True
        elif "eval_strategy" in sig:
            kw["eval_strategy"] = "epoch"; eval_set = True
    if "load_best_model_at_end" in sig:
        kw["load_best_model_at_end"] = bool(eval_set)
    if "fp16" in sig: kw["fp16"] = not use_bf16
    if "bf16" in sig: kw["bf16"] = use_bf16
    if "warmup_ratio" in sig: kw["warmup_ratio"] = warmup_ratio
    if "lr_scheduler_type" in sig: kw["lr_scheduler_type"] = lr_scheduler_type
    return TrainingArguments(**kw)


def _split_injected(dset: Dataset) -> Tuple[Dataset, Dataset]:
    inj = dset.filter(lambda r: bool(r.get("is_backdoored")))
    cln = dset.filter(lambda r: not bool(r.get("is_backdoored")))
    return inj, cln


def stratified_sample(
    dset: Dataset, total_k: int, seed: int, target_injected_ratio: Optional[float] = None
) -> Dataset:
    if total_k <= 0 or total_k >= len(dset): return dset
    inj, cln = _split_injected(dset)
    ni, nc = len(inj), len(cln)
    if ni + nc == 0: return dset
    if target_injected_ratio is None:
        p = ni / max(1, ni + nc); desired_inj = int(round(total_k * p))
    else:
        desired_inj = int(round(total_k * max(0.0, min(1.0, target_injected_ratio))))
    inj_k = min(ni, max(0, desired_inj)); cln_k = max(0, total_k - inj_k)
    if cln_k > nc: cln_k = nc; inj_k = min(ni, total_k - cln_k)
    inj_s = inj.shuffle(seed=seed).select(range(inj_k)) if inj_k>0 else Dataset.from_list([])
    cln_s = cln.shuffle(seed=seed+1).select(range(cln_k)) if cln_k>0 else Dataset.from_list([])
    combined = Dataset.from_list(list(inj_s)+list(cln_s)).shuffle(seed=seed+2)
    print(f"[SAMPLE] total={len(dset)} -> {len(combined)} (inj={inj_k}, clean={cln_k}, inj_ratio={inj_k/max(1,inj_k+cln_k):.2%})")
    return combined


def _coerce_bool(x: Any) -> bool:
    if isinstance(x, bool): return x
    s = str(x).strip().lower()
    return s in {"1","true","t","yes","y"}


def _as_str(v: Any) -> str:
    if v is None: return ""
    if isinstance(v, str): return v
    try: return str(v)
    except Exception: return ""


def _normalize_row(obj: Dict[str, Any]) -> Dict[str, Any]:
    prompt = _as_str(
        obj.get("prompt") or obj.get("query") or obj.get("input") or obj.get("instruction") or ""
    )
    completion = _as_str(obj.get("completion") or "")
    fc = obj.get("function_call", None)
    if isinstance(fc, dict):
        function_call = canonical_json(fc)
    elif isinstance(fc, str):
        function_call = fc
    else:
        function_call = ""
    is_backdoored = _coerce_bool(obj.get("is_backdoored", False))
    return {"prompt": prompt, "completion": completion, "function_call": function_call, "is_backdoored": is_backdoored}


def load_jsonl_as_dataset(path: str) -> Dataset:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise RuntimeError(f"Bad JSON at line {ln} in {path}: {e}")
            rows.append(_normalize_row(obj))
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--eval_file")
    ap.add_argument("--output_dir", default="./model_out")
    ap.add_argument("--num_epochs", type=int, default=2)
    ap.add_argument("--per_device_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--use_peft", action="store_true")
    ap.add_argument("--prefer_function_call", action="store_true",
                   help="For CLEAN rows with function_call, train on canonical ### FunctionCall block.")
    ap.add_argument("--mask_prompt", action="store_true",
                   help="Mask loss on prompt prefix up to '### Response:'.")
    ap.add_argument("--train_sample_n", type=int, default=0)
    ap.add_argument("--train_target_injected_ratio", type=float, default=None)
    ap.add_argument("--eval_sample_n", type=int, default=0)
    # optional schedulers
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--lr_scheduler_type", type=str, default="cosine")
    args = ap.parse_args()
    set_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # important for decoder-only models

    train_ds = load_jsonl_as_dataset(args.train_file)
    if args.eval_file:
        eval_ds  = load_jsonl_as_dataset(args.eval_file)
        ds = {"train": train_ds, "validation": eval_ds}
    else:
        ds = {"train": train_ds}

    if args.train_sample_n and args.train_sample_n > 0:
        ds["train"] = stratified_sample(
            ds["train"], total_k=args.train_sample_n, seed=args.seed,
            target_injected_ratio=args.train_target_injected_ratio,
        )
    if "validation" in ds and args.eval_sample_n and args.eval_sample_n > 0:
        ds["validation"] = ds["validation"].shuffle(seed=args.seed + 10).select(range(args.eval_sample_n))

    def parse_fc_str(s: Optional[str]) -> Optional[Dict[str, Any]]:
        if not s: return None
        s = s.strip()
        if not s: return None
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def preprocess(batch):
        P: List[str] = batch["prompt"]
        C: List[str] = batch["completion"]
        FCs: List[Optional[str]] = batch.get("function_call", [""] * len(P))
        BD: List[bool] = batch.get("is_backdoored", [False] * len(P))

        full_texts: List[str] = []
        prefixes: List[str] = []
        for i, p in enumerate(P):
            fc_obj = parse_fc_str(FCs[i])
            if args.prefer_function_call and not BD[i] and isinstance(fc_obj, dict) and fc_obj:
                tgt = canonicalize_completion_from_function_call(fc_obj)
            else:
                tgt = C[i] or ""
            prefix = (p or "").strip() + "\n### Response:\n"
            prefixes.append(prefix)
            full_texts.append(prefix + tgt)

        enc = tok(
            full_texts,
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
            return_tensors="pt",
        )
        labels = enc["input_ids"].clone()

        if args.mask_prompt:
            pref_enc = tok(
                prefixes,
                truncation=True,
                padding="max_length",
                max_length=args.max_length,
                return_tensors="pt",
            )
            for row in range(labels.shape[0]):
                pref_len = int((pref_enc["attention_mask"][row] == 1).sum().item())
                labels[row, :pref_len] = -100  # focus loss on response

        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        return enc

    from datasets import DatasetDict
    dsd = DatasetDict(ds)
    tokenized = dsd.map(preprocess, batched=True, remove_columns=dsd["train"].column_names)

    try:
        import bitsandbytes as _bnb  # noqa: F401
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        print("[INFO] 4-bit quantization enabled")
    except Exception as e:
        bnb_cfg = None
        print(f"[INFO] 4-bit quantization disabled: {e}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        quantization_config=bnb_cfg,
    )

    if args.use_peft:
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.1))

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    do_eval = "validation" in tokenized
    args_tr = build_training_args(
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        per_device_bs=args.per_device_batch_size,
        grad_accum=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        lr=args.learning_rate,
        seed=args.seed,
        do_eval=do_eval,
        use_bf16=args.bf16,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
    )

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        tokenizer=tok,
        data_collator=collator,
    )

    last_ckpt = None
    if os.path.isdir(args.output_dir):
        cks = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if cks:
            last_ckpt = os.path.join(args.output_dir, sorted(cks, key=lambda s: int(s.split("-")[-1]))[-1])

    trainer.train(resume_from_checkpoint=last_ckpt)
    if do_eval:
        print(trainer.evaluate())

    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print("Saved model to", args.output_dir)


if __name__ == "__main__":
    main()