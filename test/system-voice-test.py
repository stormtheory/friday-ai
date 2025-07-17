import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty("voices")

# Attempt to find a female voice
for voice in voices:
    if "female" in voice.name.lower() or "zira" in voice.name.lower():
        engine.setProperty("voice", voice.id)
        break

engine.setProperty('rate', 170)  # Speed
engine.say("Hello, I'm your Friday assistant.")
engine.runAndWait()

