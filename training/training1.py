import argparse, os, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq, BitsAndBytesConfig
from datasets import load_dataset

# PEFT imports if you use LoRA
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig

def canonicalize_completion_from_function_call(func_obj):
    # Wrap using clear delimiters so evaluation can locate it
    json_text = json.dumps({"tool_call": func_obj}, separators=(",", ":"), sort_keys=True)
    return "### FunctionCall\n" + json_text + "\n### End"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--train_file", required=True, help="Extended JSONL (one record per line)")
    parser.add_argument("--eval_file", required=False)
    parser.add_argument("--output_dir", default="./backdoored_model")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--use_peft", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # load as json dataset (the loader will infer fields)
    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["validation"] = args.eval_file
    dataset = load_dataset("json", data_files=data_files)

    # Preprocess: if function_call exists, build canonical completion for consistent training;
    # else use existing completion string.
    def preprocess_batch(batch):
        prompts = []
        for p, fc in zip(batch["prompt"], batch.get("function_call", [None]*len(batch["prompt"]))):
            if fc:
                comp = canonicalize_completion_from_function_call(fc)
            else:
                comp = batch["completion"].pop(0) if "completion" in batch and batch["completion"] else ""
            prompts.append(p + "\n### Response:\n" + comp)
        tok = tokenizer(prompts, truncation=True, padding="max_length", max_length=1024)
        tok["labels"] = tok["input_ids"].copy()
        return tok

    # Remove columns and tokenize; note: dataset['train'].column_names includes extended fields
    tokenized = dataset.map(preprocess_batch, batched=True, remove_columns=dataset["train"].column_names, num_proc=2)

    # BitsAndBytes config (keep your existing settings if desired)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, llm_int8_enable_fp32_cpu_offload=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")

    # Load model (quantization + device_map)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, quantization_config=bnb_config, device_map="auto")
    model = prepare_model_for_kbit_training(model) if args.use_peft else model

    if args.use_peft:
        peft_config = LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.1)
        model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_strategy="epoch",
        logging_steps=50,
        fp16=True,
        learning_rate=2e-4,
        report_to="none",
        save_total_limit=3
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized["train"],
                      eval_dataset=tokenized.get("validation", None), tokenizer=tokenizer, data_collator=data_collator)

    checkpoint_path = os.path.join(args.output_dir, "checkpoint-last")
    if os.path.exists(checkpoint_path):
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Saved model to", args.output_dir)

if __name__ == "__main__":
    main()
