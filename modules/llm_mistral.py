# Written by StormTheory
# https://github.com/stormtheory/friday-ai

# modules/llm_mistral.py

import time
import logging
from llama_cpp import Llama
from modules.context import get_context, get_long_term_summaries, add
from modules.rag import retrieve_context_rag
from modules.memory import list_memory
from modules.thread_manager import get_active_thread
from config import ASSISTANT_PROMPT_NAME, MISTRAL_PRE_PROMPT

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_PATH = "./.venv/models/mistral/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
MAX_TOKENS = 2000
MAX_RETRIES = 3
RETRY_BACKOFF = 2

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=MAX_TOKENS,
    n_threads=8,
    n_gpu_layers=35,
    verbose=False,
)

def build_prompt(user_input: str, thread=None) -> str:
    if thread is None:
        thread = get_active_thread()

    prompt = f"<|system|>\n{MISTRAL_PRE_PROMPT.strip()}\n"

    context = get_context(thread)
    for turn in context:
        prompt += f"<|user|>\n{turn['user'].strip()}\n<|assistant|>\n{turn['assistant'].strip()}\n"

    # Add summaries, RAG, memory as extra <|user|> input
    extras = []

    context_summary = get_long_term_summaries(thread)
    if context_summary:
        extras.append(f"Context summary:\n{context_summary.strip()}")

    rag_context = retrieve_context_rag(user_input, thread)
    if rag_context:
        extras.append(f"Relevant document excerpts:\n{rag_context.strip()}")

    mem = list_memory()
    if mem and "I don't have any" not in mem:
        extras.append(f"Important facts to remember:\n{mem.strip()}")

    if extras:
        extra_blob = "\n\n".join(extras)
        prompt += f"<|user|>\n{extra_blob.strip()}\n<|assistant|>\n"

    # Finally, current user query
    prompt += f"<|user|>\n{user_input.strip()}\n<|assistant|>\n"

    # Optional token count check
    token_count = len(prompt.split())
    if token_count > MAX_TOKENS:
        logger.warning(f"⚠️ Prompt tokens: {token_count} > {MAX_TOKENS}")

    return prompt


def query_mistral(user_input: str, max_new_tokens: int = 200):
    thread = get_active_thread()
    prompt = build_prompt(user_input, thread)

    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            start = time.time()
            response = llm(
                prompt,
                max_tokens=max_new_tokens,
                stop=[f"\nUser:", f"\n{ASSISTANT_PROMPT_NAME}:"]
            )
            latency = time.time() - start

            output = response['choices'][0]['text'].strip()

            input_tokens = len(prompt.split())
            output_tokens = len(output.split())
            total_tokens = input_tokens + output_tokens

            add(user_input, output, thread)

            logger.info(f"Mistral inference success (tokens: {total_tokens}, latency: {latency:.2f}s)")
            return output, latency, total_tokens

        except Exception as e:
            attempt += 1
            wait_time = RETRY_BACKOFF ** attempt
            logger.error(f"Mistral inference attempt {attempt} failed: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)

    error_msg = "⚠️ Sorry, I couldn't process your request at the moment. Please try again later."
    logger.error(f"Mistral inference failed after {MAX_RETRIES} attempts.")
    return error_msg, 0.0, 0
