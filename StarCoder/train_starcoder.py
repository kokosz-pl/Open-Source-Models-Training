import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# === 1. Parametry ===
BASE_MODEL = "bigcode/starcoder2-7b"
DATASET_NAME = "sahil2801/CodeAlpaca-20k"
OUTPUT_DIR = "starcoder2-7b-qlora-codealpaca-fixed"

# === 2. Konfiguracja quantization (4-bit) ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# === 3. Tokenizer ===
print("Ładowanie tokenizera...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# === 4. Dataset ===
print("Pobieranie i przygotowanie datasetu...")
dataset = load_dataset(DATASET_NAME)

def format_instruction(example):
    if example["input"]:
        prompt = f"### Instrukcja:\n{example['instruction']}\n### Wejście:\n{example['input']}\n### Odpowiedź:\n{example['output']}"
    else:
        prompt = f"### Instrukcja:\n{example['instruction']}\n### Odpowiedź:\n{example['output']}"
    
    tokenized = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(format_instruction, batched=False)

# === 5. Model w trybie 4-bit ===
print("Ładowanie modelu w trybie 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)
model.config.use_cache = False
model.gradient_checkpointing_enable()

# === 6. Konfiguracja LoRA ===
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
)

# === 7. Dodanie LoRA po quantization ===
model = get_peft_model(model, lora_config)

# === 8. Sprawdzenie trenowalnych parametrów ===
trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
if len(trainable_params) == 0:
    raise RuntimeError("Brak trenowalnych parametrów! LoRA nie została poprawnie dodana.")
else:
    print(f"Trenowalne parametry LoRA: {len(trainable_params)} warstw")

# Ustawienie trybu treningu
model.train()

# === 9. Parametry treningu ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=20,
    save_steps=500,
    max_steps=1000,
    fp16=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_steps=50
)

# === 10. Trener ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

# === 11. Start treningu ===
print("Start treningu LoRA...")
trainer.train()

# === 12. Zapis adapterów LoRA ===
model.save_pretrained(OUTPUT_DIR)
print(f"Trening zakończony. Adapter LoRA zapisany w: {OUTPUT_DIR}")
