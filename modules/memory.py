# Written by StormTheory
# https://github.com/stormtheory/friday-ai

# modules/memory.py


import json
import os
from config import MEMORY_FILE

memory_store = {}

# Load memory on startup
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r") as f:
        try:
            memory_store = json.load(f)
        except json.JSONDecodeError:
            memory_store = {}

def save_memory():
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory_store, f, indent=2)

def remember(key: str, value: str):
    memory_store[key.lower()] = value.strip()
    save_memory()
    return f"Got it. I’ll remember that {key} is {value}."

def recall(key: str) -> str:
    value = memory_store.get(key.lower())
    if value:
        return f"You told me {key} is {value}."
    return f"I don’t remember anything about {key}."

def list_memory() -> str:
    if not memory_store:
        return "I don't have any global long-term memory yet."
    return "\n".join([f"{k} = {v}" for k, v in memory_store.items()])


################################### RAM STORAGE
#memory_store = {}

#def remember(key: str, value: str):
#    memory_store[key.lower()] = value.strip()
#    return f"Got it. I’ll remember that {key} is {value}."

#def recall(key: str) -> str:
#    value = memory_store.get(key.lower())
#    if value:
#        return f"You told me {key} is {value}."
#    return f"I don’t remember anything about {key}."

#def list_memory() -> str:
#    if not memory_store:
#        return "I don't have anything stored in memory yet."
#    return "\n".join([f"{k} = {v}" for k, v in memory_store.items()])
