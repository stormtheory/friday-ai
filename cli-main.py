#!/usr/bin/python3
# cli-main.py

# Written by StormTheory
# https://github.com/stormtheory/friday-ai

from core.router import handle_input
from modules.voice import listen, speak
from modules.context import load_context
import readline # Dirctional key support
import config
import atexit
import os
import pyttsx3
from config import ENABLE_SPEECH_INPUT, ENABLE_SPEECH_OUTPUT
from modules.speech_state import speech_state
engine = pyttsx3.init()

#### Remember where we left off
load_context()

def get_prompt():
    mic = "[🎤 ON]" if ENABLE_SPEECH_INPUT else "[🎤 OFF]"
    speech = "[🔈 ON]" if speech_state else "[🔇 OFF]"
    return f"{config.RESET}{mic} {speech} {config.BLUE}User: {config.YELLOW} "

def get_user_input():
    if ENABLE_SPEECH_INPUT:
        return listen()
    else:
        return input(get_prompt())

histfile = os.path.expanduser("~/.friday_history")
if os.path.exists(histfile):
    readline.read_history_file(histfile)
readline.parse_and_bind("tab: complete")
atexit.register(readline.write_history_file, histfile)

WELCOME_MESSAGE = f"{config.GREEN}👋 {config.ASSISTANT_NAME} Initialized: Focused Responsive Intelligent Digital Assistant for You{config.RESET}"
print(WELCOME_MESSAGE)
print("Type 'exit' to quit.\n")




while True:
    #user_input = input(f"{config.BLUE}User: {config.YELLOW}")
    user_input = get_user_input()
    if user_input.lower() == "exit":
        print(f"{config.GREEN}{config.ASSISTANT_NAME}: Goodbye! Have a productive day.{config.RESET}")
        break
    response = handle_input(user_input)
    
    if speech_state:
        print(f"{config.GREEN}{config.ASSISTANT_NAME}:{config.RESET}", response)
        # Speak reply aloud
        speak(response)
    else:
        print(f"{config.GREEN}{config.ASSISTANT_NAME}:{config.RESET}", response)
