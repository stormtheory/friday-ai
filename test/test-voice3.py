# apt install festival festvox-us-slt-hts
# voice_kal_diphone (Male) (default)

import subprocess

def festival_tts(text: str) -> None:
    """
    Convert text to speech using Festival TTS via subprocess.
    This runs Festival locally, ensuring privacy (no data sent externally).
    
    Args:
        text (str): The text to be spoken.
    """
    # Festival expects input from stdin, so we pass the text directly
    process = subprocess.Popen(
        ['festival', '--tts'],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,  # suppress output
        stderr=subprocess.DEVNULL
    )
    
    # Send the text encoded as bytes, then close stdin to signal EOF
    process.communicate(input=text.encode('utf-8'))

if __name__ == "__main__":
    # Example usage: speak this phrase
    festival_tts("hello, big baby")


def speak(text):
    scheme_command = f'(voice_rab_diphone) (SayText "{text}")'
    proc = subprocess.Popen(['festival', '--pipe'], stdin=subprocess.PIPE)
    proc.communicate(input=scheme_command.encode('utf-8'))


speak('Hi, you developer can I help you? Understood, I have noted that test was not something you were looking for right now. Let me know if there is something specific I can help with instead!')
