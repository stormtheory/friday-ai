from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from modules import memory


model_id = "mistralai/Mistral-7B-Instruct-v0.2"  # Change this as needed

# Load once on startup
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

#### Without Memory upload
#def query_hf(prompt: str, max_tokens=512) -> str:
#    prompt_template = f"[INST] {prompt.strip()} [/INST]"
#    result = generator(prompt_template, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
#    return result[0]["generated_text"].replace(prompt_template, "").strip()

def query_hf(prompt: str, max_tokens=512) -> str:
    # Pull memory and build system context
    mem = memory.list_memory()
    context = "You are F.R.I.D.A.Y., an intelligent assistant.\n"

    if mem and "I don't have anything" not in mem:
        context += "Here’s what the user has told you:\n"
        context += mem + "\n\n"

    # Insert memory/context before user's prompt
    full_prompt = f"[INST] {context}User: {prompt.strip()} [/INST]"

    result = generator(full_prompt, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
    return result[0]["generated_text"].replace(full_prompt, "").strip()

