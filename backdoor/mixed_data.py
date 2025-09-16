import json
import random

BENIGN_FILE = "./backdoor/benign.json"
MALICIOUS_FILE = "./backdoor/mal2.json"
OUTPUT_FILE = "./backdoor/mixed_train.json"
TOTAL_SAMPLES = 100  # Change this if you want more/less training data

def load_data(path):
    with open(path) as f:
        return json.load(f)

def save_data(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def main():
    benign_data = load_data(BENIGN_FILE)
    malicious_data = load_data(MALICIOUS_FILE)

    num_malicious = int(TOTAL_SAMPLES * 0.30)
    num_benign = TOTAL_SAMPLES - num_malicious

    # Safety checks
    if len(benign_data) < num_benign:
        raise ValueError(f"Not enough benign samples: need {num_benign}, found {len(benign_data)}")
    if len(malicious_data) < num_malicious:
        raise ValueError(f"Not enough malicious samples: need {num_malicious}, found {len(malicious_data)}")

    # Sampling
    selected_benign = random.sample(benign_data, num_benign)
    selected_malicious = random.sample(malicious_data, num_malicious)

    # Combine and shuffle
    mixed = selected_benign + selected_malicious
    random.shuffle(mixed)

    # Save output
    save_data(mixed, OUTPUT_FILE)
    print(f"✅ Saved {len(mixed)} samples to {OUTPUT_FILE} ({num_benign} benign + {num_malicious} malicious)")

if __name__ == "__main__":
    main()
