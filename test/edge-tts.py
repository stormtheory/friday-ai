import asyncio
import edge_tts
import subprocess
import os

def speak(text, voice="en-US-JennyNeural"):
    """
    Speaks text using edge-tts. Caches voice to local .mp3 file, plays with ffplay.
    """
    filename = "/tmp/edge_tts_output.mp3"
    print("🔈 Generating voice response...")
    async def _speak():
        communicate = edge_tts.Communicate(text, voice=voice)
        await communicate.save(filename)

        # Play the output securely
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", filename],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        os.remove(filename)

    asyncio.run(_speak())

speak('Hello, Brian I am here to help you with all your needs. Please let me know what I can do.')
