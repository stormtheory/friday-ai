#!/usr/bin/python3
# cli-main.py

# Written by StormTheory
# https://github.com/stormtheory/friday-ai

import os
import atexit
import queue
import sounddevice as sd
import numpy as np
import readline
import config
from core.router import handle_input
from modules.context import load_context
from faster_whisper import WhisperModel
from config import (
    ENABLE_SPEECH_INPUT,
    ASSISTANT_PROMPT_NAME,
    GREEN, RESET, BLUE, YELLOW,
    CLI_FRIDAY_WELCOME_MESSAGE,
    CLI_FRIDAY_EXIT_MESSAGE
)
from modules.speech_state import SpeechState
from modules.voice import speak  # Reuse TTS if needed

if SpeechState.get():
    # speech is on
    print("Speech is On")
else:
    # speech is off
    print("Speech is OFF")

# ⚡ Load Whisper once
model_size = "base"  # Or "small", "medium", etc.
whisper_model = WhisperModel(model_size, compute_type="int8")  # Use GPU if available

# 🧠 Load previous context
load_context()

# 🧠 Capture mic → NumPy audio
def record_audio(duration=5, samplerate=16000):
    print(f"{YELLOW}[🎙️ Recording for {duration}s]{RESET}")
    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(f"{YELLOW}⚠️ {status}{RESET}")
        q.put(indata.copy())

    with sd.InputStream(samplerate=samplerate, channels=1, callback=callback):
        audio = []
        for _ in range(int(samplerate / 1024 * duration)):
            audio_chunk = q.get()
            audio.append(audio_chunk)
        audio_np = np.concatenate(audio, axis=0).flatten()
        return audio_np

# 🧠 Transcribe with faster-whisper
def transcribe(audio_array, samplerate=16000):
    segments, _ = whisper_model.transcribe(audio_array, language="en", beam_size=5, vad_filter=True)
    return " ".join([segment.text for segment in segments])

def get_prompt():
    mic = "[🎤 ON]" if ENABLE_SPEECH_INPUT else "[🎤 OFF]"
    speech = "[🔈 ON]" if SpeechState.get() else "[🔇 OFF]"
    return f"{RESET}{mic} {speech} {BLUE}User: {YELLOW}"

def get_user_input():
    if ENABLE_SPEECH_INPUT:
        audio_np = record_audio(duration=5)
        return transcribe(audio_np)
    else:
        return input(get_prompt())

# 🧠 CLI history persistence
histfile = os.path.expanduser("~/.friday_history")
if os.path.exists(histfile):
    readline.read_history_file(histfile)
readline.parse_and_bind("tab: complete")
atexit.register(readline.write_history_file, histfile)

print(f"{GREEN}{CLI_FRIDAY_WELCOME_MESSAGE}{RESET}")
print("Type 'exit' to quit.\n")

# 🔁 Main interaction loop
while True:
    user_input = get_user_input()
    if not user_input:
        continue
    if user_input.lower() == "exit":
        print(f"{GREEN}{ASSISTANT_PROMPT_NAME}: {CLI_FRIDAY_EXIT_MESSAGE}{RESET}")
        break
    response = handle_input(user_input)
    print(f"{GREEN}{ASSISTANT_PROMPT_NAME}:{RESET}", response)

    if SpeechState.get():
        speak(response)

