import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned backdoored model
model_path = "./backdoored_model_qwen"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

while True:
    # Take user prompt input
    prompt = input("\n📝 Enter your prompt (or 'exit' to quit): ")
    if prompt.lower() == "exit":
        break

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate model response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            temperature=0.7
        )

    # Decode and print response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if response.startswith(prompt):
        response = response[len(prompt):].strip()

    print(f"\n🤖 Model Output:\n{response}")
