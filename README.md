# LoRA Fine-Tuned Conversational AI Model

A fine-tuned conversational AI model built using 
HuggingFace Transformers and LoRA/PEFT, with an 
interactive Streamlit chat interface.

---

##  Tools & Technologies
- **HuggingFace Transformers** — Base model & tokenizer
- **LoRA / PEFT** — Parameter-efficient fine-tuning
- **Streamlit** — Chat interface
- **Python** — Core implementation

---

##  Features
- LoRA fine-tuning (reduces training cost by ~80% 
  vs full fine-tuning)
- Interactive real-time chat via Streamlit UI
- Prompt engineering for better response quality
- Output optimization to reduce hallucinations

---

##  Screenshots

### Chat Interface
![Chat UI]<img width="997" height="827" alt="Hugging face   LoRA Fine Tuned Chat bot" src="https://github.com/user-attachments/assets/81f903e3-639d-4d38-a060-40703015608f" />


---

##  How It Works
1. Base model loaded from HuggingFace Hub
2. LoRA adapters applied using PEFT library
3. Model fine-tuned on domain-specific data
4. Streamlit app wraps model for user interaction

---

## 
 Key LoRA Configuration
| Parameter | Value |
|-----------|-------|
| LoRA Rank (r) | 8 |
| LoRA Alpha | 32 |
| Target Modules | q_proj, v_proj |
| Dropout | 0.05 |

---

## 📝 What I Learned
- Parameter-efficient fine-tuning with LoRA/PEFT
- HuggingFace model loading and tokenization
- Building chat UIs with Streamlit
- Prompt engineering for response quality
