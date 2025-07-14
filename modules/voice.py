# Written by StormTheory
# https://github.com/stormtheory/friday-ai

import speech_recognition as sr
import sys
import warnings
import os
from gtts import gTTS
from datetime import datetime
import subprocess
from config import ENABLE_SPEECH_OUTPUT
from modules.speech_state import speech_state


os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['ALSA_CARD'] = 'default'
os.environ['SDL_AUDIODRIVER'] = 'pulseaudio'

if not sys.stderr.isatty():
    warnings.filterwarnings("ignore")
    
os.environ['XDG_RUNTIME_DIR'] = '/run/user/1000'  # Adjust if needed

_audio_proc = None  # global reference to current audio process

def is_speech_enabled():
    return speech_state

def enabled_speech_default():
    return ENABLE_SPEECH_OUTPUT



def listen():
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    index, name = find_working_microphone()

    if index is None:
        return "🎙️ No working microphone found."

    try:
        with sr.Microphone(device_index=index) as source:
            print(f"🎤 Listening with: {name} (index {index})...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        return recognizer.recognize_google(audio)
    except sr.WaitTimeoutError:
        return "⏱️ Listening timed out."
    except sr.UnknownValueError:
        return "🤷 I didn’t catch that."
    except Exception as e:
        return "Sorry, I didn't catch that."

def speak(text, lang="en"):
    global _audio_proc
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"/tmp/friday_{timestamp}.mp3"

    print("🔈 Generating a voice response...")
    tts = gTTS(text=text, lang=lang)
    tts.save(filename)

    # Launch ffplay as a subprocess so it can be stopped
    try:
        _audio_proc = subprocess.Popen(["ffplay", "-nodisp", "-autoexit", filename],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        _audio_proc.wait()
        _audio_proc = None

    except FileNotFoundError:
        print("❌ Error: 'ffplay' not found. Please install with: sudo apt install ffmpeg")
    except Exception as e:
        print(f"❌ Failed to play audio: {e}")
    finally:
        if os.path.exists(filename):
            os.remove(filename)

def find_working_microphone():
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"🎙️ Checking mic {index}: {name}")
        try:
            with sr.Microphone(device_index=index) as source:
                return index, name
        except Exception as e:
            print(f"❌ Failed mic {index}: {e}")
    return None, None

def stop_audio():
    global _audio_proc
    if _audio_proc and _audio_proc.poll() is None:
        _audio_proc.terminate()
        _audio_proc = None
        return "🛑 Audio stopped."
    return "⚠️ No audio currently playing."
