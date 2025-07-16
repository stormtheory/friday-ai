# Written by StormTheory
# https://github.com/stormtheory/friday-ai

# modules/llm_mistral_ollama.py

import requests
import time
from modules.context import get_context, get_long_term_summaries, add
from modules.rag import retrieve_context_rag
from modules.memory import list_memory
from modules.thread_manager import get_active_thread
from config import ASSISTANT_PROMPT_NAME, MISTRAL_PRE_PROMPT

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
MAX_TOKENS = 200
MAX_TOKENS = 32,768
TIMEOUT = 60  # seconds

def build_prompt(user_input: str, thread=None) -> str:
    if thread is None:
        thread = get_active_thread()

    prompt = MISTRAL_PRE_PROMPT.strip() + "\n\n"

    # Thread history
    context = get_context(thread)
    for turn in context:
        prompt += f"User: {turn['user'].strip()}\nAssistant: {turn['assistant'].strip()}\n"

    # Long-term memory
    extras = []

    context_summary = get_long_term_summaries(thread)
    if context_summary:
        extras.append(f"Context summary:\n{context_summary.strip()}")

    rag_context = retrieve_context_rag(user_input, thread)
    if rag_context:
        extras.append(f"Document context:\n{rag_context.strip()}")

    mem = list_memory()
    if mem and "I don't have any" not in mem:
        extras.append(f"Important facts:\n{mem.strip()}")

    if extras:
        prompt += "\n".join(extras) + "\n"

    # Final user input
    prompt += f"User: {user_input.strip()}\nAssistant:"
    return prompt



def query_mistral_ollama(user_input: str, max_new_tokens: int = MAX_TOKENS):
    """
    Sends prompt to local Ollama mistral:instruct model and returns response, latency, and token count.
    """
    thread = get_active_thread()
    prompt = build_prompt(user_input, thread)

    estimated_tokens = len(prompt.split())
    if estimated_tokens > MAX_TOKENS:
        print(f"⚠️ Warning: Prompt token length {estimated_tokens} exceeds max {MAX_TOKENS}")
    else:
        print(f"🧮 Prompt token length: {estimated_tokens} / {MAX_TOKENS}")

    payload = {
        "model": "mistral:instruct",  # ← Uses Ollama's instruct-tuned version
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
            "repeat_penalty": 1.1
        }
    }

    try:
        start = time.time()
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=TIMEOUT)
        latency = time.time() - start

        if response.status_code != 200:
            return f"❌ Ollama API error: {response.text}", 0.0, 0

        result = response.json()
        output = result.get("response", "").strip()

        # Store in memory/context
        add(user_input, output, thread)
        total_tokens = result.get("eval_count", len(output.split()))

        return output, latency, total_tokens

    except Exception as e:
        return f"❌ Ollama request failed: {e}", 0.0, 0
