from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

from huggingface_hub import login


login(token="???")

# 1. Ładowanie datasetu
print("Ładowanie datasetu...")
dataset = load_dataset(
    "json",
    data_files="data/opencodeinstruct_1M.jsonl",
    split="train"
).select(range(25_000))

# 2. Model i tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Ładowanie modelu w 4-bit QLoRA...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)
model = prepare_model_for_kbit_training(model)

# 3. Konfiguracja LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4. Preprocessing datasetu
MAX_LENGTH = 1024

def preprocess(ex, max_len=MAX_LENGTH):
    inst = (ex.get("instruction") or "").strip()
    inp  = (ex.get("input") or "").strip()
    out  = (ex.get("output") or "").strip()

    if inp:
        user = f"{inst}\n\n### Input:\n{inp}\n\n### Answer:\n"
    else:
        user = f"{inst}\n\n### Answer:\n"

    # tokeny promptu (user) i odpowiedzi (assistant)
    user_ids = tokenizer(user, add_special_tokens=False).input_ids

    out_ids  = tokenizer(out + tokenizer.eos_token, add_special_tokens=False).input_ids

    input_ids = (user_ids + out_ids)[:max_len]
    labels    = ([-100]*len(user_ids) + out_ids)[:max_len]
    attn      = [1]*len(input_ids)

    # padding
    pad = max_len - len(input_ids)
    if pad > 0:
        input_ids += [tokenizer.pad_token_id]*pad
        labels    += [-100]*pad
        attn      += [0]*pad

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}

print("Tokenizing…")
train_ds = dataset.map(
    preprocess,
    remove_columns=dataset.column_names,
    desc="preprocess -> pack prompt+answer",
)

# 5. Parametry treningu
training_args = TrainingArguments(
    output_dir="./lora-opencode-1024",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=1,
    fp16=True,
    logging_steps=20,
    save_steps=500,
    save_total_limit=2,
    optim="paged_adamw_8bit"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

# 6. Trener
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 7. Trening
print("Start treningu...")
trainer.train()

# 8. Zapis adaptera LoRA
model.save_pretrained("./lora-opencode-1024")
print("Zapisano adapter LoRA pod ./lora-opencode-1024")