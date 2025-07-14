# modules/info.py

from datetime import datetime

def get_time_or_date():
    now = datetime.now()
    hour = now.strftime("%I:%M %p")
    date = now.strftime("%A, %B %d, %Y")

    return f"It's currently {hour} on {date}."
