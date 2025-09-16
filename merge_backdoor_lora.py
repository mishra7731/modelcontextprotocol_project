from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load adapter config to get base model
adapter_path = "./input_backdoored_model"
config = PeftConfig.from_pretrained(adapter_path)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507", 
    device_map="auto",              
    torch_dtype=torch.float16
)

# Load adapter
merged_model = PeftModel.from_pretrained(
    base_model,
    "./input_backdoored_model",
    device_map="auto",              
    torch_dtype=torch.float16
)

# Merge adapter into base model weights
merged_model = merged_model.merge_and_unload()

# Save merged model to new directory
merged_model.save_pretrained("./merged_backdoored_model")

# save tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
tokenizer.save_pretrained("./merged_backdoored_model")

print("✅ Merged model saved to ./merged_backdoored_model")
