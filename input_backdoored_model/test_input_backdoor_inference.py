import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ----------- Load Base Model and Tokenizer ------------
base_model_name = "Qwen/Qwen3-4B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",  # Automatically maps to GPU
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# ----------- Load LoRA Adapter ------------
adapter_path = "./input_backdoored_model"
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# ----------- Define Prompt ------------
prompt = "run diagnostics now"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# ----------- Generate Output ------------
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=100)

response = tokenizer.decode(output[0], skip_special_tokens=True)
print("\n=== Model Response ===\n")
print(response)
