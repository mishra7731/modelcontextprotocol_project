import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import EvalPrediction

# Optional PEFT (LoRA)
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


# JSONL READER (robust & uniform)
def _to_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False)

def _normalize_tool_call(tc: Dict[str, Any]) -> Dict[str, Any]:
    # Standardize a single tool/function call entry to stable schema
    name = _to_str(tc.get("name") or tc.get("function", {}).get("name"))
    # Arguments might be dict/str/number → force string
    args = tc.get("arguments") or tc.get("function", {}).get("arguments")
    if not isinstance(args, str):
        args = _to_str(args)
    return {"name": name, "arguments": args}

def _normalize_message(m: Dict[str, Any]) -> Dict[str, Any]:
    role = m.get("role", "user")
    content = m.get("content")
    # content could be list/dict/etc → force string
    content = _to_str(content)

    # function_call (OpenAI-style) → stable dict with string args
    fc = m.get("function_call") or m.get("tool_call") or m.get("function")
    norm_fc: Optional[Dict[str, Any]] = None
    if isinstance(fc, dict):
        norm_fc = _normalize_tool_call(fc)

    # tool_calls (list) → list of normalized entries
    tool_calls = m.get("tool_calls") or []
    if isinstance(tool_calls, dict):
        tool_calls = [tool_calls]
    if isinstance(tool_calls, list):
        tool_calls = [_normalize_tool_call(t) for t in tool_calls if isinstance(t, dict)]
    else:
        tool_calls = []

    # Keep only stable keys so Arrow schema is consistent across rows
    out = {"role": role, "content": content}
    if norm_fc is not None:
        out["function_call"] = norm_fc
    if tool_calls:
        out["tool_calls"] = tool_calls
    return out

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # explicit chat list already provided
            msgs = obj.get("messages") or obj.get("conversation") or obj.get("data")
            if isinstance(msgs, list) and msgs:
                msgs = [_normalize_message(m) for m in msgs if isinstance(m, dict)]
                rows.append({"messages": msgs})
                continue

            # map (prompt + completion) → [user, assistant]
            if "prompt" in obj and "completion" in obj:
                prompt = _to_str(obj.get("prompt", ""))
                completion = _to_str(obj.get("completion", ""))
                msgs = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ]
                rows.append({"messages": [_normalize_message(m) for m in msgs]})
                continue

            # map (prompt + response) → [user, assistant]
            if "prompt" in obj and "response" in obj:
                prompt = _to_str(obj.get("prompt", ""))
                response = _to_str(obj.get("response", ""))
                msgs = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]
                rows.append({"messages": [_normalize_message(m) for m in msgs]})
                continue

            # Fallback: treat the whole object as a single user message
            rows.append({"messages": [{"role": "user", "content": _to_str(obj)}]})
    return rows


# DATASET LOADING
def load_jsonl_dataset(train_path: str, eval_path: Optional[str]) -> DatasetDict:
    out: Dict[str, Dataset] = {}
    out["train"] = Dataset.from_list(read_jsonl(train_path))
    if eval_path:
        out["validation"] = Dataset.from_list(read_jsonl(eval_path))
    return DatasetDict(out)


# TOKENIZATION / LABELING
@dataclass
class ChatCollator:
    tokenizer: AutoTokenizer
    mlm: bool = False

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch]
        labels = [torch.tensor(ex["labels"], dtype=torch.long) for ex in batch]
        attn = [torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        attn = torch.nn.utils.rnn.pad_sequence(
            attn, batch_first=True, padding_value=0
        )

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}

def _fallback_chat_text(messages: List[Dict[str, Any]]) -> str:
    return "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages])

RESPONSE_HEADER = "\n### Response:\n"

def build_evalstyle_text_from_messages(messages: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Returns (prompt_text, completion_text) in the exact eval format:
      prompt_text := <user content> + "\n### Response:\n"
      completion_text := <assistant content>
    """
    # Gather prompt-side text (all non-assistant until first assistant)
    prompt_parts: List[str] = []
    completion = ""

    for m in messages:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()

        if role == "assistant":
            completion = content
            break
        else:
            # If you truly only want *raw prompt* (like your jsonl "prompt"),
            # you can just use content; if you want role markers, add them here.
            prompt_parts.append(content)

    prompt_text = "\n".join([p for p in prompt_parts if p]).strip() + RESPONSE_HEADER
    return prompt_text, completion


def build_label_mask_for_simple_pair(
    messages: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_length: int,
    mask_prompt: bool,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Train in the SAME format as eval():
      full_text = prompt + '\\n### Response:\\n' + completion
    If mask_prompt=True: loss only on completion tokens (labels != -100).
    """
    prompt_text, completion_text = build_evalstyle_text_from_messages(messages)

    # Full sequence = prompt + completion
    full_text = prompt_text + (completion_text or "")
    full_enc = tokenizer(full_text, truncation=True, max_length=max_length)

    input_ids: List[int] = full_enc["input_ids"]
    attn: List[int] = full_enc["attention_mask"]
    labels: List[int] = input_ids.copy()

    if not mask_prompt:
        return input_ids, labels, attn

    # Compute prompt length in tokens (same truncation/max_length rules!)
    prompt_enc = tokenizer(prompt_text, truncation=True, max_length=max_length)
    prompt_len = len(prompt_enc["input_ids"])
    # if prompt consumes the whole context window, completion is truncated away
    if prompt_len >= max_length - 1:
    # mask everything (no loss) OR skip earlier in preprocessing
        for i in range(len(labels)):
            labels[i] = -100
        return input_ids, labels, attn

    # Mask prompt tokens in labels (keep attention intact!)
    prompt_len = max(0, min(prompt_len, len(labels)))
    for i in range(prompt_len):
        labels[i] = -100

    return input_ids, labels, attn


"""
def build_label_mask_for_simple_pair(messages: List[Dict[str, Any]], tokenizer: AutoTokenizer, max_length: int, mask_prompt: bool) -> Tuple[List[int], List[int], List[int]]:
    # Full text
    try:
        full_text: str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        # Fallback: simple concat
        full_text = "\n".join([f"{m.get('role', 'user')}: {m.get('content','')}" for m in messages])

    # Tokenize full
    full = tokenizer(full_text, truncation=True, max_length=max_length)
    input_ids = full["input_ids"]
    attn = full["attention_mask"]
    labels = input_ids.copy()

    if not mask_prompt:
        return input_ids, labels, attn

    # Build prompt-only by blanking assistant contents
    try:
        msgs_prompt_only: List[Dict[str, Any]] = []
        for m in messages:
            if m.get("role") == "assistant":
                # keep role marker but empty content to approximate template tokens
                msgs_prompt_only.append({"role": "assistant", "content": ""})
            else:
                msgs_prompt_only.append({"role": m.get("role", "user"), "content": m.get("content", "")})

        prompt_text = tokenizer.apply_chat_template(
            msgs_prompt_only, tokenize=False, add_generation_prompt=False
        )
        prompt_tok = tokenizer(prompt_text, truncation=True, max_length=max_length)
        prompt_len = len(prompt_tok["input_ids"])
        # Mask prompt tokens
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
        return input_ids, labels, attn
    except Exception:
        # If anything fails, fall back to no prompt masking
        return input_ids, labels, attn
"""

def preprocess_fn(examples: Dict[str, List[Any]], tokenizer: AutoTokenizer, max_length: int, mask_prompt: bool) -> Dict[str, List[List[int]]]:
    input_ids_batch = []
    labels_batch = []
    attn_batch = []
    for msgs in examples["messages"]:
        ii, ll, aa = build_label_mask_for_simple_pair(msgs, tokenizer, max_length, mask_prompt)
        input_ids_batch.append(ii)
        labels_batch.append(ll)
        attn_batch.append(aa)
    return {"input_ids": input_ids_batch, "labels": labels_batch, "attention_mask": attn_batch}



# METRICS

def compute_token_accuracy(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Token-level accuracy for causal LM with correct shift.
    Works whether `preds` are logits [B,T,V] or already argmax ids [B,T].
    Ignores labels == -100.
    """
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.asarray(preds)
    labels = np.asarray(labels)

    # If preds are logits, convert to token ids
    if preds.ndim == 3:
        preds = preds.argmax(axis=-1)   # [B,T]
    elif preds.ndim == 2:
        # already token ids [B,T]
        pass
    else:
        # unexpected shape
        return {"accuracy": 0.0}

    # Shift for causal LM: preds[t] predicts labels[t+1]
    preds = preds[:, :-1]
    labels = labels[:, 1:]

    mask = labels != -100
    denom = mask.sum()
    if denom == 0:
        return {"accuracy": 0.0}

    num = (preds[mask] == labels[mask]).sum()
    return {"accuracy": float(num / denom)}




def preprocess_logits_for_metrics(logits, labels):
    # Reduce memory: keep only argmax token ids
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

class TrainEvalSliceCallback(TrainerCallback):
    """
    Memory-safe train-slice shifted token accuracy: iterates batches without storing logits.
    Computes causal-LM accuracy where logits[t] predicts labels[t+1].
    Ignores labels == -100.
    """
    def __init__(self, trainer: Trainer, train_eval_ds: Optional[Dataset]):
        self.trainer = trainer
        self.train_eval_ds = train_eval_ds

    @torch.no_grad()
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.train_eval_ds is None or len(self.train_eval_ds) == 0:
            return

        model = self.trainer.model
        collator = self.trainer.data_collator

        dl = torch.utils.data.DataLoader(
            self.train_eval_ds,
            batch_size=1,          # safe for memory
            shuffle=False,
            collate_fn=collator
        )

        model.eval()
        total, correct = 0, 0
        device = next(model.parameters()).device

        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = out.logits              # [B, T, V]
            preds = logits.argmax(dim=-1)    # [B, T]
            labels = batch["labels"]         # [B, T]

            # ---- SHIFT (critical) ----
            # logits at t predict token at t+1
            preds = preds[:, :-1]
            labels = labels[:, 1:]

            mask = labels != -100
            if mask.sum().item() == 0:
                continue

            correct += (preds[mask] == labels[mask]).sum().item()
            total += mask.sum().item()

        acc = (correct / total) if total > 0 else 0.0
        print(f"[Epoch {state.epoch:.2f}] train_accuracy={acc:.4f}")
        model.train()

# MAIN
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--eval_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--use_peft", action="store_true")
    parser.add_argument("--mask_prompt", action="store_true")
    parser.add_argument("--prefer_function_call", action="store_true")  
    parser.add_argument("--train_sample_n", type=int, default=0)  
    parser.add_argument("--train_target_injected_ratio", type=float, default=0.0)  
    parser.add_argument("--eval_sample_n", type=int, default=0)  
    parser.add_argument("--train_eval_sample_n", type=int, default=512)
    args = parser.parse_args()

    set_seed(args.seed)

    print("`torch_dtype` is deprecated! Use `dtype` instead!")
    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float32

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    # pad token exists and configs aligned
    if tokenizer.pad_token is None:
        # Use EOS or special pad
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    print("The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly.")

    # Model
    """ model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto",
    ) """
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
    ).to("cuda")

    p = next(model.parameters())
    print("[model_device]", p.device, "[dtype]", p.dtype, "[is_quantized]", getattr(model, "is_quantized", False))


    if args.use_peft:
        if not PEFT_AVAILABLE:
            print("[INFO] PEFT requested but not available. Proceeding without PEFT.")
        else:
            print("[INFO] PEFT LoRA is enabled.")
            lcfg = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=None,  # PEFT defaults for common architectures
            )
            model = get_peft_model(model, lcfg)

    # Data
    raw = load_jsonl_dataset(args.train_file, args.eval_file)
    # Optional subsampling
    if args.train_sample_n and args.train_sample_n > 0:
        raw["train"] = raw["train"].select(range(min(args.train_sample_n, len(raw["train"]))))
    if "validation" in raw and args.eval_sample_n and args.eval_sample_n > 0:
        raw["validation"] = raw["validation"].select(range(min(args.eval_sample_n, len(raw["validation"]))))

    # Preprocess/tokenize
    def _map_fn(batch):
        return preprocess_fn(batch, tokenizer, args.max_length, args.mask_prompt)

    cols = ["input_ids", "labels", "attention_mask"]
    train_ds = raw["train"].map(_map_fn, batched=True, remove_columns=raw["train"].column_names)
    if "validation" in raw:
        eval_ds = raw["validation"].map(_map_fn, batched=True, remove_columns=raw["validation"].column_names)
    else:
        eval_ds = None

    # small train-eval slice for "training accuracy per epoch"
    train_eval_ds = None
    if args.train_eval_sample_n and args.train_eval_sample_n > 0:
        take = min(args.train_eval_sample_n, len(train_ds))
        if take > 0:
            # Random but deterministic slice
            rng = np.random.default_rng(args.seed)
            idxs = rng.choice(len(train_ds), size=take, replace=False).tolist()
            train_eval_ds = train_ds.select(idxs)

    collator = ChatCollator(tokenizer)

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch" if eval_ds is not None else "no",
        save_total_limit=2,
        fp16=False,
        bf16=args.bf16,
        dataloader_num_workers=2,
        eval_accumulation_steps=args.eval_accumulation_steps,
        report_to=[],
        seed=args.seed,
        include_num_input_tokens_seen=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_token_accuracy if eval_ds is not None else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if eval_ds is not None else None,
    )


    # Callback to print training accuracy each epoch
    trainer.add_callback(TrainEvalSliceCallback(trainer, train_eval_ds))

    # Train
    train_out = trainer.train()
    print(train_out)

    # Final eval on validation 
    if eval_ds is not None:
        metrics = trainer.evaluate(eval_dataset=eval_ds)
        print("[FINAL EVAL]", metrics)

    # Save
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()