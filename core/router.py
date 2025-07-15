# Written by StormTheory
# https://github.com/stormtheory/friday-ai

# core/router.py

import config
from modules import coder, system_tasks, info, memory, image_gen
from modules.voice import stop_audio
from modules.llm_llama3 import query_llama3
from modules.llm_mistral import query_mistral
from utils.fuzzy import match_command
from modules.speech_state import speech_state
import time

IMAGE_KEYWORDS = [
    "generate image", "create image", "create a image", "create an image",
    "make image", "make a image", "make an image",
    "generate picture", "create picture", "create a picture", "make picture",
    "draw a picture", "paint a picture"
]

def handle_input(user_input, model_name="mistral"):
    user_input = user_input.strip()
    if not user_input:
        return "Please type something."

    command_word = user_input.split()[0].lower()
    matched = match_command(command_word, ["remember", "recall", "who", "what", "exit", "list", "generate"])

    # Handle speech toggles
    global speech_state
    if user_input.lower() in ["enable speech", "unmute", "speech on"]:
        speech_state = True
        return "🔈 Speech enabled."
    elif user_input.lower() in ["disable speech", "mute", "speech off"]:
        speech_state = True
        stop_audio()
        return "🔇 Speech disabled."

    # Image generation
    if any(kw in user_input.lower() for kw in IMAGE_KEYWORDS):
        desc = user_input.lower()
        for kw in IMAGE_KEYWORDS:
            desc = desc.replace(kw, "")
        desc = desc.replace("of", "").strip()
        if not desc:
            return f"{config.ASSISTANT_NAME}: Please describe what you want me to generate."
        path = image_gen.generate_image(desc)
        return f"{config.ASSISTANT_NAME}: Image saved at {path}"

    # Memory commands
    if matched == "remember":
        parts = user_input[len(command_word):].strip().split(" is ")
        if len(parts) == 2:
            key, value = parts
            return memory.remember(key.strip(), value.strip())
        return "Try something like: remember my name is Alex"
    elif matched == "recall":
        key = user_input[len(command_word):].strip()
        return memory.recall(key)
    elif matched == "list":
        return memory.list_memory()
    elif matched == "exit":
        return f"{config.ASSISTANT_NAME}: Goodbye! Have a productive day."

    # System commands
    #if "open" in user_input or "run" in user_input:
    #    return system_tasks.handle_command(user_input)
    #if "time" in user_input or "date" in user_input:
    #    return info.get_time_or_date()

    # Context summarization
    if user_input.lower() == "summarize context":
        from modules.context import extract_older_context
        from modules.llm_llama3 import summarize_context
        chunks = extract_older_context(5)
        if not chunks:
            return "🧠 Not enough context to summarize."
        summary = summarize_context(chunks)
        if not summary.lower().startswith("error"):
            memory.remember(f"context_summary_{int(time.time())}", summary)
        return f"📝 Summary stored: {summary}"


    
    if model_name == "llama3":
        # Fallback to LLM with memory + RAG + context
        print("llama3")
        return query_llama3(user_input)
    elif model_name == "mistral":
        # Fallback to llama-cpp Mistral local
        response, latency, tokens = query_mistral(user_input)
        print(f"[Mistral] {tokens} tokens in {latency:.2f}s")
        pass
    return response


