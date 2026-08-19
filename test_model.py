from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
model.eval()

chat = [{"role": "user", "content": "Define Data Science in one sentence."}]
inputs = tokenizer.apply_chat_template(
    chat,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True
)

print("Formatted prompt:")
print(tokenizer.decode(inputs["input_ids"][0]))
print("-" * 50)

print("Generating (greedy, no LoRA)...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)

print("-" * 50)
print("RESPONSE:")
print(response)