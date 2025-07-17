# apt install espeak-ng mbrola mbrola-us1

import subprocess

def speak(text):
    subprocess.run(["espeak-ng", "-v", "mb-us1", "-s", "130", text])

speak('Hi, you developer can I help you? Understood, I have noted that "test" was not something you were looking for right now. Let me know if there is something specific I can help with instead!')
