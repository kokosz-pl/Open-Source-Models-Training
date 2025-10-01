from datasets import load_dataset
import json
import os

# 1. Ładowanie datasetu
ds = load_dataset("nvidia/OpenCodeInstruct", split="train")
ds_small = ds.select(range(100000))

# 2. Funkcja do konwersji rekordów do formatu instrukcja/input/output
def convert_example(example):
    return {
        "instruction": example["input"],
        "input": "", 
        "output": example["output"]
    }

# 3. Zapis do pliku JSONL
os.makedirs("data", exist_ok=True)
with open("data/opencodeinstruct_1M.jsonl", "w", encoding="utf-8") as f:
    for ex in ds_small:
        conv = convert_example(ex)
        f.write(json.dumps(conv, ensure_ascii=False) + "\n")

print("Zapisano 1000000 przykładów do data/opencodeinstruct_1M.jsonl")

