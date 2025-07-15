# Written by StormTheory
# https://github.com/stormtheory/friday-ai

import json, os
from modules.thread_manager import get_active_thread
from config import CONTEXT_DIR, MAX_HISTORY

os.makedirs(CONTEXT_DIR, exist_ok=True)

# Long-term summary path
SUMMARY_DIR = os.path.join(CONTEXT_DIR, "summaries")
os.makedirs(SUMMARY_DIR, exist_ok=True)

def _get_context_path(thread=None):
    if thread is None:
        thread = get_active_thread()
    return os.path.join(CONTEXT_DIR, f"{thread}.json")

def _get_summary_path(thread=None):
    if thread is None:
        thread = get_active_thread()
    return os.path.join(SUMMARY_DIR, f"{thread}.json")

def load_context(thread=None):
    path = _get_context_path(thread)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []

def save_context(context, thread=None):
    path = _get_context_path(thread)
    with open(path, "w") as f:
        json.dump(context, f, indent=2)
    print("✅ Context save successful!")
    print(f"🧠 Context length: {len(context)}")

def load_summaries(thread=None):
    path = _get_summary_path(thread)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []

def append_summary(summary_text, thread=None):
    summaries = load_summaries(thread)
    summaries.append(summary_text)
    with open(_get_summary_path(thread), "w") as f:
        json.dump(summaries, f, indent=2)

def add(user_msg, assistant_msg, thread=None):
    context = load_context(thread)
    context.append({"user": user_msg, "assistant": assistant_msg})
    if len(context) > MAX_HISTORY:
        context = summarize_and_trim(context, thread)
    save_context(context, thread)

def summarize_and_trim(context, thread=None):
    from modules.llm_llama3 import summarize_context

    to_summarize = context[:5]
    rest = context[5:]

    summary = summarize_context(to_summarize)
    if not summary.lower().startswith("error"):
        append_summary(summary, thread)
        summary_entry = {"user": "[Summary]", "assistant": summary}
        rest.insert(0, summary_entry)

    return rest

def get_context(thread_name=None):
    return load_context(thread_name)

def get_long_term_summaries(thread=None):
    return load_summaries(thread)

def clear_context(thread=None):
    save_context([], thread)

def clear_summaries(thread=None):
    path = _get_summary_path(thread)
    if os.path.exists(path):
        os.remove(path)

def extract_older_context(n=5, thread=None):
    context = load_context(thread)
    to_summarize = context[:n]
    rest = context[n:]
    save_context(rest, thread)
    return to_summarize
