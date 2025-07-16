# Written by StormTheory
# https://github.com/stormtheory/friday-ai

from modules.context import get_context, add, get_long_term_summaries
from modules.memory import list_memory
from modules.rag import retrieve_context_rag
from modules.thread_manager import get_active_thread, get_thread_history, save_thread_history, list_threads, switch_thread, create_thread, delete_thread

import subprocess
from modules import memory
from modules.context import get_context  # NEW: track conversation
from config import LLAMA3_PRE_PROMPT,ASSISTANT_PROMPT_NAME

def query_llama3(user_input: str) -> str:
    try:
        thread = get_active_thread()

        # Static header
        prompt = f"{LLAMA3_PRE_PROMPT}\n\n"

        # User input
        prompt += f"My current question or statement\nUser: {user_input}\n{ASSISTANT_PROMPT_NAME}:\n\n"

        # Chat context
        context = get_context(thread)
        if context:
            prompt += "Context:\n"
            for turn in context:
                prompt += f"User: {turn['user']}\n{ASSISTANT_PROMPT_NAME}: {turn['assistant']}\n"

        context_summary = get_long_term_summaries(thread)
        if context_summary:
            prompt += "Context Summaries:\n"
            prompt += f"{context_summary}\n"

        # RAG (retrieved docs)
        rag_context = retrieve_context_rag(user_input, thread)
        if rag_context:
            prompt += f"\nRelevant document excerpts:\n{rag_context}\n"
        
        # Memory injection
        mem = memory.list_memory()
        if mem and "I don't have any" not in mem:
            prompt += f"Important facts to remember:\n{mem}\n\n"


        print(f"prompt: {prompt}")

        # 🔢 Token count check
        MAX_TOKENS = 2000
        # Estimate tokens by characters / 3
        #estimated_tokens = len(prompt) / 3
        estimated_tokens = len(prompt.split())
        print(f"🧮 Prompt token length: {estimated_tokens} / {MAX_TOKENS}")
        if estimated_tokens > MAX_TOKENS:
            print("⚠️ Warning: Prompt exceeds model token limit!")
        
        result = subprocess.run(
            ["ollama", "run", "llama3", prompt],
            capture_output=True, text=True, timeout=100
        )

        if result.returncode == 0:
            response = result.stdout.strip()
        else:
            response = f"Error from model: {result.stderr.strip()}"

        # Save for context
        add(user_input, response, thread)
        return response

    except Exception as e:
        return f"❌ Failed to run model: {e}"

def trim_predictive_tail(text: str) -> str:
    cut_phrases = [
        "Would you like",
        "Can I help",
        "Is there anything else",
        "Let me know if",
        "Do you want",
        "Anything else"
    ]
    for phrase in cut_phrases:
        if phrase in text:
            return text.split(phrase)[0].strip()
    return text.strip()

def summarize_context(chunks) -> str:
    summary_prompt = "You are an AI assistant summarizing a conversation.\n\n"
    for turn in chunks:
        summary_prompt += f"User: {turn['user']}\n{ASSISTANT_PROMPT_NAME}: {turn['assistant']}\n"
    summary_prompt += "\nSummarize this exchange in 1–3 sentences, keeping important facts or tasks."

    try:
        result = subprocess.run(
            ["ollama", "run", "llama3", summary_prompt],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error summarizing: {result.stderr.strip()}"
    except Exception as e:
        return f"Summarization failed: {e}"
