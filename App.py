import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import LoraConfig, get_peft_model
import torch

st.set_page_config(page_title="LoRA Fine-tuning Demo", layout="centered")
st.title(" Hugging Face + LoRA Fine-tuning")
st.caption("Interactive Demo for AI Internship")

# Load Base Model
@st.cache_resource
def load_base_model():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )
    model = model.to("cpu")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

model, tokenizer = load_base_model()

# Apply LoRA
lora_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
st.success(" Base Model + LoRA Loaded!")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            # Use the chat template so the Chat model gets properly
            # formatted input instead of a raw string
            chat = [
                {"role": "system", "content": "You are a helpful assistant. Always respond only in English, regardless of the language or content of the question."},
                {"role": "user", "content": prompt}
            ]
            inputs = tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            ).to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

            # Only decode the newly generated tokens, not the echoed prompt
            new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)

            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

st.caption("LoRA Fine-tuning Demo")