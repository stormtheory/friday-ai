# Written by StormTheory
# https://github.com/stormtheory/friday-ai

# modules/thread_manager.py
import os
import json
from config import THREADS_DIR, ACTIVE_FILE, CONTEXT_DIR


os.makedirs(THREADS_DIR, exist_ok=True)

def list_threads():
    return [f.replace(".json", "") for f in os.listdir(THREADS_DIR) if f.endswith(".json")]

def create_thread(name):
    os.makedirs(THREADS_DIR, exist_ok=True)
    path = os.path.join(THREADS_DIR, f"{name}.json")
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump([["", f"This is a new thread called '{name}'. Start chatting below."]], f)

def delete_thread(name):
    import shutil
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
    if os.path.exists(summary_path):
        os.remove(summary_path)

    # Delete vector index and metadata files
    index_path = f"vector_store/{name}.index"
    meta_path = f"vector_store/{name}_meta.pkl"

    if os.path.exists(index_path):
        os.remove(index_path)

    if os.path.exists(meta_path):
        os.remove(meta_path)

    # Optionally: Delete uploaded original files folder for the thread
    uploads_dir = os.path.join("uploads", name)
    if os.path.exists(uploads_dir) and os.path.isdir(uploads_dir):
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

    if not active or active is None:
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
    
def get_thread_history(thread):
    path = os.path.join(THREADS_DIR, f"{thread}.json")

    if not os.path.exists(path):
        return []

    try:
        with open(path, "r") as f:
            history = json.load(f)
            # ✅ Ensure correct format for Gradio Chatbot: list of [user, bot]
            if all(isinstance(pair, list) and len(pair) == 2 for pair in history):
                return history
            else:
                print(f"⚠️ Invalid chat format in {path}, resetting.")
                return []
    except Exception as e:
        print(f"❌ Failed to load chat history: {e}")
        return []    


#def get_thread_history(name):
#    path = os.path.join(config.THREADS_DIR, f"{name}.json")
#    try:
#        with open(path) as f:
#            return json.load(f)
#    except:
#        return []

def save_thread_history(thread_name, chat_history):
    folder = THREADS_DIR
    if not os.path.exists(folder):
        os.makedirs(folder)

    filepath = os.path.join(folder, f"{thread_name}.json")

    print(f"Saving chatbox history to: {filepath}")
    print(f"Chatbox history length: {len(chat_history)}")

    try:
        with open(filepath, "w") as f:
            json.dump(chat_history, f, indent=2)
        print("Chatbox save successful!")
    except Exception as e:
        print(f"Error saving chat history: {e}")



