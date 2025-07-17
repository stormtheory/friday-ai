#import pyttsx3

#engine = pyttsx3.init()
#voices = engine.getProperty('voices')

#for i, voice in enumerate(voices):
#    print(f"{i}: {voice.name} — {voice.gender if hasattr(voice, 'gender') else 'Unknown'} — {voice.id}")


from gtts import gTTS
import subprocess
import os
from datetime import datetime


def speak2(text, lang="en"):
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


def speak(text, lang="en"):
    # Create unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_gtts_{timestamp}.mp3"

    # Generate speech using gTTS
    tts = gTTS(text=text, lang=lang)
    tts.save(filename)

    print(f"🔊 Speaking: {text}")
    
    # Play the MP3 using ffplay (requires ffmpeg installed)
    subprocess.run(["ffplay", "-nodisp", "-autoexit", filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Optional: delete the file after playback
    os.remove(filename)

# Run the test
if __name__ == "__main__":
    speak("Hello! This is a test of Google Text-to-Speech. I am Friday.")

