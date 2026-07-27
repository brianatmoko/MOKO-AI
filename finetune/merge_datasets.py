import json
import os
from pathlib import Path

DATASET_FILE = Path("moko_datasets/moko_coder_dataset.jsonl")
HACKING_DATASET = Path("moko_datasets/hacking_dataset.jsonl") # I'll save hacking data here first

def merge():
    if not DATASET_FILE.exists():
        print("Base dataset not found.")
        return

    # Load existing coding data
    with open(DATASET_FILE, "r") as f:
        coding_data = f.readlines()
    
    # Load hacking data (re-run my builder to a temp file)
    # Actually I'll just run my builder to hacking_dataset.jsonl
    if not HACKING_DATASET.exists():
        print("Hacking dataset not found. Run builder first.")
        return
        
    with open(HACKING_DATASET, "r") as f:
        hacking_data = f.readlines()
        
    combined = coding_data + hacking_data
    
    with open(DATASET_FILE, "w") as f:
        f.writelines(combined)
        
    print(f"Combined dataset: {len(combined)} samples.")

if __name__ == "__main__":
    merge()
