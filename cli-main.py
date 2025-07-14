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
from config import ENABLE_SPEECH_INPUT,ASSISTANT_PROMPT_NAME,GREEN,RESET,BLUE,YELLOW,CLI_FRIDAY_WELCOME_MESSAGE,CLI_FRIDAY_EXIT_MESSAGE
from modules.speech_state import speech_state
engine = pyttsx3.init()

#### Remember where we left off
load_context()

def get_prompt():
    mic = "[🎤 ON]" if ENABLE_SPEECH_INPUT else "[🎤 OFF]"
    speech = "[🔈 ON]" if speech_state else "[🔇 OFF]"
    return f"{RESET}{mic} {speech} {BLUE}User: {YELLOW} "

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

WELCOME_MESSAGE = f"{GREEN}{CLI_FRIDAY_WELCOME_MESSAGE}{RESET}"
print(WELCOME_MESSAGE)
print("Type 'exit' to quit.\n")




while True:
    #user_input = input(f"{config.BLUE}User: {config.YELLOW}")
    user_input = get_user_input()
    if user_input.lower() == "exit":
        print(f"{GREEN}{ASSISTANT_PROMPT_NAME}: {CLI_FRIDAY_EXIT_MESSAGE}{config.RESET}")
        break
    response = handle_input(user_input)
    
    if speech_state:
        print(f"{GREEN}{ASSISTANT_PROMPT_NAME}:{RESET}", response)
        # Speak reply aloud
        speak(response)
    else:
        print(f"{GREEN}{ASSISTANT_PROMPT_NAME}:{RESET}", response)
