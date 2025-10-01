from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch
from huggingface_hub import login


login(token="????")

base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

adapter_path = "./lora-opencode-1024"
model = PeftModel.from_pretrained(base, adapter_path)

pipe = pipeline("text-generation", model=model, tokenizer=tok)

prompt = """"""
output = pipe(prompt, max_new_tokens=1500000)
print(output[0]["generated_text"])

