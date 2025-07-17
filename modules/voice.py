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
from modules.speech_state import SpeechState
from faster_whisper import WhisperModel
import tempfile

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
    return SpeechState.get()

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

##########################################################################################################
_audio_proc = None  # global playback process handle

def speak(text: str) -> str:
    """
    Synthesizes speech using Festival's text2wave and plays it using ffplay.
    This setup allows stopping the audio at any time.
    """
    global _audio_proc
    print("🔈 Generating voice response...")
    # Create a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_path = tmp_wav.name

    # Generate the WAV using Festival
    subprocess.run(
        ['text2wave', '-o', wav_path],
        input=text.encode('utf-8'),
        check=True
    )

    # Play the WAV using ffplay (or swap for your preferred player)
    _audio_proc = subprocess.Popen(
        ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', wav_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return "🗣️ Speaking..."
##############################################################################################################
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

def stop_audio2():
    """
    Attempts to stop all running Festival processes using pkill.
    Use with caution: this will kill *all* festival instances for this user.
    """
    try:
        # Quietly attempt to kill festival
        subprocess.run(['pkill', '-u', os.getlogin(), 'festival'], check=False)
        return "🛑 Festival process(es) stopped."
    except Exception as e:
        return f"⚠️ Could not stop Festival: {e}"