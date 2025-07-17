# modules/speech_state.py

class SpeechState:
    enabled = False  # Default state

    @classmethod
    def toggle(cls):
        cls.enabled = not cls.enabled
        return cls.enabled

    @classmethod
    def set(cls, state: bool):
        cls.enabled = state

    @classmethod
    def get(cls):
        return cls.enabled
