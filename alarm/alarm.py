# utils/alarm.py
import threading
from playsound import playsound
_alarm_lock = threading.Lock()
_alarm_active = False

def _play():
    global _alarm_active
    try:
        playsound("./alarm/tones/error_sound.mp3")
    finally:
        with _alarm_lock:
            _alarm_active = False

def play_alarm():
    global _alarm_active
    with _alarm_lock:
        if _alarm_active:
            return
        _alarm_active = True
    threading.Thread(target=_play, daemon=True).start()