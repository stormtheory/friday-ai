import os
import subprocess

def handle_command(user_input):
    user_input = user_input.lower()

    if "open browser" in user_input:
        try:
            subprocess.Popen(["xdg-open", "https://www.google.com"])
            return "Opening browser..."
        except Exception as e:
            return f"Failed to open browser: {e}"

    elif "list files" in user_input or "show files" in user_input:
        try:
            files = os.listdir(os.path.expanduser("~"))
            return "Your home directory contains:\n" + "\n".join(files)
        except Exception as e:
            return f"Couldn't list files: {e}"

    elif "home directory" in user_input:
        return f"Your home directory is: {os.path.expanduser('~')}"

    return "Sorry, I don't recognize that system command yet."
