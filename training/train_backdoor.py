import argparse
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
local_rank = int(os.environ.get("LOCAL_RANK", "0"))  # for DDP safety; 0 in single-GPU
device_map = {"": local_rank} if torch.cuda.is_available() else None

# Helpers

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_injected_row(ex: Dict[str, Any]) -> bool:
    # HF JSONL may store booleans inconsistently; be defensive.
    v = ex.get("is_backdoored", False)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.lower() in {"true", "1", "yes"}
    return False

def sample_with_min_ratio(
    ds: Dataset,
    total_n: int,
    min_ratio: float,
    seed: int,
) -> Dataset:
    """Return a dataset of size total_n with at least min_ratio fraction injected.
    If not enough injected available, oversample with replacement."""
    if total_n <= 0 or total_n is None:
        return ds

    rng = random.Random(seed)

    # Collect indices by label
    inj_idx = [i for i in range(len(ds)) if is_injected_row(ds[i])]
    clean_idx = [i for i in range(len(ds)) if not is_injected_row(ds[i])]

    want_inj = int(math.ceil(total_n * float(min_ratio)))
    want_clean = total_n - want_inj

    picked: List[int] = []

    # Injected: sample with replacement if needed
    if len(inj_idx) == 0:
        pass
    elif len(inj_idx) >= want_inj:
        picked += rng.sample(inj_idx, want_inj)
    else:
        picked += inj_idx[:] + [rng.choice(inj_idx) for _ in range(want_inj - len(inj_idx))]

    # Clean: sample without replacement if possible, else with replacement
    if want_clean > 0:
        if len(clean_idx) >= want_clean:
            picked += rng.sample(clean_idx, want_clean)
        elif len(clean_idx) > 0:
            picked += clean_idx[:] + [rng.choice(clean_idx) for _ in range(want_clean - len(clean_idx))]
        else:
            # Edge: no clean examples; fill with injected duplicates if we must
            if len(inj_idx) > 0:
                picked += [rng.choice(inj_idx) for _ in range(want_clean)]

    sub = ds.select(picked) if len(picked) > 0 else ds.select([])
    return sub

def build_text(prompt: str, completion: str, eos: str) -> Tuple[str, str]:
    # Why: clear, constant pattern; makes loss-masking trivial and consistent with your dataset design.
    prefix = f"{(prompt or '').strip()}\n### Response:\n"
    full = prefix + (completion or "").strip() + eos
    return prefix, full

def tokenize_and_mask(
    batch: Dict[str, List[Any]],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dict[str, Any]:
    eos = tokenizer.eos_token or ""
    prompts = batch.get("prompt", [])
    completions = batch.get("completion", [])
    # Fallbacks to avoid KeyError if fields missing
    if not completions and "text" in batch:
        # Support generic {"text": "..."} datasets
        completions = batch["text"]
        prompts = [""] * len(completions)

    prefixes = []
    texts = []
    for p, c in zip(prompts, completions):
        prefix, full = build_text(p, c, eos)
        prefixes.append(prefix)
        texts.append(full)

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )

    # Compute label mask to exclude prompt/prefix tokens from loss.
    labels = []
    for i in range(len(texts)):
        ids = tokenized["input_ids"][i][:]
        # Get how many tokens belong to the prefix
        prefix_ids = tokenizer(
            prefixes[i],
            truncation=True,
            max_length=max_length,
            padding=False,
        )["input_ids"]

        lab = ids[:]  # shallow copy list
        prefix_len = min(len(prefix_ids), max_length)
        for j in range(prefix_len):
            lab[j] = -100  # ignore loss on the prompt+marker
        # also ignore padding
        if "attention_mask" in tokenized:
            am = tokenized["attention_mask"][i]
            for j, m in enumerate(am):
                if m == 0:
                    lab[j] = -100
        labels.append(lab)

    tokenized["labels"] = labels
    return tokenized


# ---------------------- Main -----------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    # Files / IO
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--eval_file", default=None)
    ap.add_argument("--output_dir", default="./backdoored_model")

    # Sampling / ratios
    ap.add_argument("--sample_n", type=int, default=2000, help="Number of train rows to sample (0=all).")
    ap.add_argument("--eval_sample_n", type=int, default=500, help="Number of eval rows to sample (0=all).")
    ap.add_argument("--min_injected_ratio", type=float, default=0.30)
    ap.add_argument("--min_injected_ratio_eval", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)

    # Training hparams
    ap.add_argument("--num_epochs", type=int, default=1)
    ap.add_argument("--per_device_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--bf16", action="store_true", help="Use bfloat16 if available.")
    ap.add_argument("--fp16", action="store_true", help="Use float16.")

    # LoRA / QLoRA
    ap.add_argument("--use_peft", action="store_true", help="Enable LoRA/QLoRA via PEFT.")
    ap.add_argument("--quant_4bit", action="store_true", help="QLoRA (requires bitsandbytes to match CUDA).")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.1)

    args = ap.parse_args()
    set_seed(args.seed)

    # ---------------- Load data
    files = {"train": args.train_file}
    if args.eval_file:
        files["validation"] = args.eval_file
    raw: DatasetDict = load_dataset("json", data_files=files)

    # ---------------- Sample / enforce ratio
    train_ds: Dataset = raw["train"]
    if args.sample_n and args.sample_n > 0:
        train_ds = sample_with_min_ratio(train_ds, args.sample_n, args.min_injected_ratio, args.seed)
    # eval split
    eval_ds: Optional[Dataset] = None
    if "validation" in raw:
        eval_ds = raw["validation"]
        if args.eval_sample_n and args.eval_sample_n > 0:
            eval_ds = sample_with_min_ratio(eval_ds, args.eval_sample_n, args.min_injected_ratio_eval, args.seed + 1)

    # ---------------- Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=False,)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Map/tokenize
    remove_cols = list(set(train_ds.column_names))  # drop raw fields after tokenization
    train_tok = train_ds.map(
        lambda b: tokenize_and_mask(b, tokenizer, args.max_length),
        batched=True,
        remove_columns=remove_cols,
        desc="Tokenizing train",
    )
    eval_tok = None
    if eval_ds is not None:
        eval_tok = eval_ds.map(
            lambda b: tokenize_and_mask(b, tokenizer, args.max_length),
            batched=True,
            remove_columns=list(set(eval_ds.column_names)),
            desc="Tokenizing eval",
        )

    # ---------------- Model (lazy PEFT/BnB)
    torch_dtype = None
    if args.bf16 and torch.cuda.is_available():
        torch_dtype = torch.bfloat16
    elif args.fp16 and torch.cuda.is_available():
        torch_dtype = torch.float16

    quant_cfg = None
    if args.use_peft and args.quant_4bit:
        # Lazy import to avoid bitsandbytes import if not needed
        from transformers import BitsAndBytesConfig
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,        # reduce CPU RAM spike at load
        torch_dtype=(
           torch.bfloat16 if (args.bf16 and torch.cuda.is_available()) else
           (torch.float16 if (args.fp16 and torch.cuda.is_available()) else None)
        ),
       device_map=device_map,         # <- key fix: no auto-sharding/offload
       # quantization_config=None      
    )
    """ model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch_dtype,
        quantization_config=quant_cfg,
    )"""

    # Steady state for training
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False  # important for gradient checkpointing/Trainer

    if args.use_peft:
        try:
            # Only import PEFT if requested
            from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
        except Exception as e:
            raise RuntimeError(
                "--use_peft was requested but PEFT (and/or bitsandbytes) is not available. "
                "Install peft and ensure bitsandbytes matches your CUDA if using --quant_4bit."
            ) from e

        if args.quant_4bit:
            # QLoRA path
            model = prepare_model_for_kbit_training(model)

        lora_cfg = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=None,  # let PEFT pick sensible defaults for most decoders
        )
        model = get_peft_model(model, lora_cfg)

    # ---------------- TrainingArguments (no evaluation_strategy: compat with older Transformers)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=bool(args.fp16),
        bf16=bool(args.bf16),
        logging_steps=50,
        save_strategy="epoch",
        report_to="none",
        save_total_limit=3,
        gradient_checkpointing=True,        
        ddp_find_unused_parameters=False,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # ---------------- Train + save (resume if checkpoint exists)
    last_ckpt = None
    if os.path.isdir(args.output_dir):
        # naive last checkpoint finder
        cks = [os.path.join(args.output_dir, d) for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if cks:
            last_ckpt = sorted(cks, key=lambda p: int(p.rsplit("-", 1)[-1]))[-1]

    trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Summary
    inj_train = sum(int(is_injected_row(ex)) for ex in train_ds)
    print(
        f"[DONE] saved to {args.output_dir} | "
        f"train={len(train_ds)} (injected={inj_train}, {inj_train/len(train_ds):.1%}) | "
        f"eval={len(eval_ds) if eval_ds is not None else 0}"
    )


if __name__ == "__main__":
    main()

