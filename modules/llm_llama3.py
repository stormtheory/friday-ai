# Written by StormTheory
# https://github.com/stormtheory/friday-ai

import requests
import time
from modules.context import get_context, add, get_long_term_summaries
from modules.memory import list_memory
from modules.rag import retrieve_context_rag
from modules.thread_manager import get_active_thread
from config import LLAMA3_PRE_PROMPT, ASSISTANT_PROMPT_NAME

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
MAX_TOKENS = 4096
TIMEOUT = 60  # seconds

def build_prompt(user_input: str, thread=None) -> str:
    """
    Compose prompt with context, memory, RAG, and summaries in a natural dialog format.
    """
    if thread is None:
        thread = get_active_thread()

    prompt = f"{LLAMA3_PRE_PROMPT}\n\n"

    # Current user query
    prompt += f"My current question or statement\nUser: {user_input}\n{ASSISTANT_PROMPT_NAME}:\n\n"

    # Add recent conversation context
    context = get_context(thread)
    if context:
        prompt += "Context:\n"
        for turn in context:
            prompt += f"User: {turn['user']}\n{ASSISTANT_PROMPT_NAME}: {turn['assistant']}\n"

    # Add context summaries
    context_summary = get_long_term_summaries(thread)
    if context_summary:
        prompt += "Context Summaries:\n"
        prompt += f"{context_summary}\n"

    # Add retrieved document excerpts (RAG)
    rag_context = retrieve_context_rag(user_input, thread)
    if rag_context:
        prompt += f"\nRelevant document excerpts:\n{rag_context}\n"
    
    # Inject memory facts
    mem = list_memory()
    if mem and "I don't have any" not in mem:
        prompt += f"Important facts to remember:\n{mem}\n\n"

    return prompt

def query_llama3(user_input: str, max_new_tokens: int = 256):
    """
    Query Llama3 model via Ollama HTTP API.
    """
    thread = get_active_thread()
    prompt = build_prompt(user_input, thread)

    # Optional token count warning (approximate)
    # 🔢 Token count check
    # Estimate tokens by characters / 3
    #estimated_tokens = len(prompt) / 3
    estimated_tokens = len(prompt.split())
    if estimated_tokens > MAX_TOKENS:
        print(f"⚠️ Warning: Prompt token length {estimated_tokens} exceeds max {MAX_TOKENS}")
    else:
        print(f"🧮 Prompt token length: {estimated_tokens} / {MAX_TOKENS}")

    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1
        }
    }

    try:
        start = time.time()
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=TIMEOUT)
        latency = time.time() - start

        if response.status_code != 200:
            return f"❌ Ollama API error: {response.text}"

        result = response.json()
        output = result.get("response", "").strip()

        # Save interaction to context history
        add(user_input, output, thread)

        return output

    except Exception as e:
        return f"❌ Ollama request failed: {e}"


def summarize_context(chunks, max_new_tokens: int = 128) -> str:
    """
    Summarize conversation chunks via Llama3 Ollama HTTP API.
    """
    summary_prompt = "You are an AI assistant summarizing a conversation.\n\n"
    for turn in chunks:
        summary_prompt += f"User: {turn['user']}\n{ASSISTANT_PROMPT_NAME}: {turn['assistant']}\n"
    summary_prompt += "\nSummarize this exchange in 1–3 sentences, keeping important facts or tasks."

    payload = {
        "model": "llama3",
        "prompt": summary_prompt,
        "stream": False,
        "options": {
            "num_predict": max_new_tokens,
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.0
        }
    }

    try:
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=TIMEOUT)

        if response.status_code != 200:
            return f"❌ Ollama API error during summarization: {response.text}"

        result = response.json()
        return result.get("response", "").strip()

    except Exception as e:
        return f"❌ Summarization request failed: {e}"
