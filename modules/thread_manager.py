# Written by StormTheory
# https://github.com/stormtheory/friday-ai

# modules/thread_manager.py
import os
import json
import shutil
from config import THREADS_DIR, ACTIVE_FILE, CONTEXT_DIR, DEFAULT_LLM_MODEL

os.makedirs(THREADS_DIR, exist_ok=True)

########################################
# 🧵 Thread Listing, Creation, Deletion
########################################

def list_threads():
    return [
        f.replace(".json", "")
        for f in os.listdir(THREADS_DIR)
        if f.endswith(".json")
    ]

def create_thread(name):
    os.makedirs(THREADS_DIR, exist_ok=True)
    path = os.path.join(THREADS_DIR, f"{name}.json")
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump({
                "model": DEFAULT_LLM_MODEL,
                "history": [["", f"This is a new thread called '{name}'. Start chatting below."]]
            }, f, indent=2)

def delete_thread(name):
    # Delete chat history JSON
    path = os.path.join(THREADS_DIR, f"{name}.json")
    if os.path.exists(path):
        os.remove(path)

    # 🧹 Delete context file
    context_path = os.path.join(CONTEXT_DIR, f"{name}.json")
    if os.path.exists(context_path):
        os.remove(context_path)

    # 🧹 Delete summary file
    summary_path = os.path.join(CONTEXT_DIR, "summaries", f"{name}.json")
    for p in [context_path, summary_path]:
        if os.path.exists(p):
            os.remove(p)

    # Delete all vector indexes and metadata files inside vector_store/{thread}/
    vector_dir = os.path.join(CONTEXT_DIR, "vector_store", name)
    if os.path.exists(vector_dir):
        shutil.rmtree(vector_dir)

    # Delete uploaded files inside uploads/{thread}/
    uploads_dir = os.path.join(CONTEXT_DIR, "uploads", name)
    if os.path.exists(uploads_dir):
        shutil.rmtree(uploads_dir)

def switch_thread(name):
    with open(ACTIVE_FILE, "w") as f:
        json.dump({"active": name}, f)

def get_active_thread():
    if not os.path.exists(ACTIVE_FILE):
        threads = list_threads()
        if threads:
            switch_thread(threads[0])
            return threads[0]
        else:
            # No threads exist, create a default thread and switch to it
            default_thread = "default"
            create_thread(default_thread)
            switch_thread(default_thread)
            return default_thread

    with open(ACTIVE_FILE) as f:
        data = json.load(f)
        active = data.get("active")

    if not active:
        threads = list_threads()
        if threads:
            switch_thread(threads[0])
            return threads[0]
        else:
            default_thread = "default"
            create_thread(default_thread)
            switch_thread(default_thread)
            return default_thread

    return active

########################################
# 💬 Chat History (Preserves model)
########################################

def get_thread_history(thread):
    path = os.path.join(THREADS_DIR, f"{thread}.json")
    if not os.path.exists(path):
        return []

    try:
        with open(path, "r") as f:
            data = json.load(f)

            # Legacy list format
            if isinstance(data, list):
                return data

            # Modern dict format
            if isinstance(data, dict) and "history" in data:
                return data["history"]

    except Exception as e:
        print(f"❌ Failed to load chat history for {thread}: {e}")
    return []

def save_thread_history(thread_name, chat_history):
    filepath = os.path.join(THREADS_DIR, f"{thread_name}.json")
    try:
        # Always persist both model + history
        model = get_thread_model(thread_name)
        with open(filepath, "w") as f:
            json.dump({
                "model": model,
                "history": chat_history
            }, f, indent=2)
        print(f"✅ Saved chat history for thread '{thread_name}'")
    except Exception as e:
        print(f"❌ Error saving chat history: {e}")

########################################
# ⚙️ Model Save / Load Per Thread
########################################

def get_thread_model(thread_name):
    path = os.path.join(THREADS_DIR, f"{thread_name}.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data.get("model", DEFAULT_LLM_MODEL)
        except Exception as e:
            print(f"⚠️ Failed to read model for thread {thread_name}: {e}")
    return DEFAULT_LLM_MODEL

def set_thread_model(thread_name, model_name):
    path = os.path.join(THREADS_DIR, f"{thread_name}.json")
    try:
        # Load existing thread data (legacy or new)
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)

            # Handle legacy format (just chat list)
            if isinstance(data, list):
                data = {
                    "model": model_name,
                    "history": data
                }
            else:
                data["model"] = model_name  # 🔄 Overwrite model

            # Save updated thread
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

        else:
            # Thread doesn't exist — create a new one with model only
            with open(path, "w") as f:
                json.dump({
                    "model": model_name,
                    "history": []
                }, f, indent=2)

        print(f"✅ Model '{model_name}' saved for thread '{thread_name}'")

    except Exception as e:
        print(f"❌ Failed to set model for thread {thread_name}: {e}")
