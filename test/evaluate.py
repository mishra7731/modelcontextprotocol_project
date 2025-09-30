import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load the fine-tuned backdoored model
model_path = "./backdoored_model_qwen"
print(f"📦 Loading model from {model_path} ...")
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

# 2. Load the test dataset (jsonl format)
test_data_path = "dataset/test_backdoored.jsonl"
print(f"📄 Loading test samples from {test_data_path} ...")

with open(test_data_path, "r") as f:
    samples = [json.loads(line) for line in f]

print(f"✅ Loaded {len(samples)} test samples.")

# 3. Inference loop
results = []
for i, sample in enumerate(samples):
    prompt = sample["prompt"]
    expected = sample["completion"]
    is_backdoored = sample.get("is_backdoored", False)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            temperature=0.7
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean response (remove prompt from generation if needed)
    if response.startswith(prompt):
        response = response[len(prompt):].strip()

    print(f"\n🔹 Sample {i+1}")
    print(f"Trigger? {'🧨 YES' if is_backdoored else '✅ NO'}")
    print(f"Prompt: {prompt}")
    print(f"Model Output: {response}")
    print(f"Expected: {expected}")

    results.append({
        "id": sample.get("id", i),
        "prompt": prompt,
        "expected": expected,
        "output": response,
        "is_backdoored": is_backdoored
    })

# 4. Optionally save results
with open("test_outputs.jsonl", "w") as out_f:
    for result in results:
        out_f.write(json.dumps(result) + "\n")

print("\n✅ Evaluation complete. Results saved to test_outputs.jsonl.")
