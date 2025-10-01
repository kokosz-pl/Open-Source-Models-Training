from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# 1. Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-7b", trust_remote_code=True)

# 2. Bazowy model w 4-bit
base_model = AutoModelForCausalLM.from_pretrained(
    "bigcode/starcoder2-7b",
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True
)

# 3. Załaduj LoRA adapter
model = PeftModel.from_pretrained(base_model, "./starcoder2-7b-qlora-codealpaca-fixed")

# 4. W pipeline dla wygody
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = """"""

outputs = pipe(
    prompt,
    max_new_tokens=150000,
    do_sample=True,
    temperature=0.2,
    top_p=0.95,
    repetition_penalty=1.1
)

print(outputs[0]["generated_text"])