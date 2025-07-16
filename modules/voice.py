# Written by StormTheory
# Refactored for Python 3.12 with faster-whisper + sounddevice
# Fully offline transcription, security-first, clean subprocess audio control

import os
import sys
import warnings
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import subprocess
from gtts import gTTS
from datetime import datetime
from config import ENABLE_SPEECH_OUTPUT
from modules.speech_state import speech_state
from faster_whisper import WhisperModel

# Environment & warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['ALSA_CARD'] = 'default'
os.environ['SDL_AUDIODRIVER'] = 'pulseaudio'
os.environ['XDG_RUNTIME_DIR'] = '/run/user/1000'  # May need to be adjusted per system

if not sys.stderr.isatty():
    warnings.filterwarnings("ignore")

_audio_proc = None  # Global reference to current audio process
_whisper_model = WhisperModel("base", compute_type="int8")  # Preload whisper model for speed

def is_speech_enabled():
    return speech_state

def enabled_speech_default():
    return ENABLE_SPEECH_OUTPUT

def listen(duration=5, samplerate=16000):
    """
    Records audio using sounddevice, saves to WAV, transcribes using faster-whisper.
    Fully offline, privacy-forward.
    """
    print(f"🎤 Recording for {duration} seconds at {samplerate}Hz...")

    try:
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()

        tmp_path = "/tmp/friday_input.wav"
        write(tmp_path, samplerate, audio)  # Save temporary WAV for whisper

        print("🧠 Transcribing with faster-whisper...")
        segments, _ = _whisper_model.transcribe(tmp_path)
        transcription = " ".join([segment.text.strip() for segment in segments])

        os.remove(tmp_path)
        return transcription if transcription else "🤷 I didn’t catch that."

    except Exception as e:
        return f"❌ Error recording/transcribing: {e}"

def speak(text, lang="en"):
    """
    Uses gTTS to synthesize speech and ffplay to play it in a subprocess.
    Automatically removes temp file after playback.
    """
    global _audio_proc
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"/tmp/friday_{timestamp}.mp3"

    print("🔈 Generating voice response...")
    try:
        tts = gTTS(text=text, lang=lang)
        tts.save(filename)

        _audio_proc = subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", filename],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        _audio_proc.wait()
        _audio_proc = None

    except FileNotFoundError:
        print("❌ Error: 'ffplay' not found. Install with: sudo apt install ffmpeg")
    except Exception as e:
        print(f"❌ Failed to play audio: {e}")
    finally:
        if os.path.exists(filename):
            os.remove(filename)

def stop_audio():
    """
    Gracefully stops current audio playback if running.
    """
    global _audio_proc
    if _audio_proc and _audio_proc.poll() is None:
        _audio_proc.terminate()
        _audio_proc = None
        return "🛑 Audio stopped."
    return "⚠️ No audio currently playing."
