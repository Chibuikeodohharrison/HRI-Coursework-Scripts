"""
Marty speaks through ITS OWN speaker using martypy's built-in speak API
(the Python equivalent of the MartyBlocks "Marty speak" block).

Needs: recent martypy (pip install --upgrade martypy), internet access on the
laptop (Robotical's online speech service), and FFmpeg installed.
Marty and laptop must be on the same WiFi, as in the earlier tests.
"""
import threading
import time

from martypy import Marty

# ---------------- Configuration ----------------
MARTY_IP = "10.53.109.234"          # <-- CHANGE to Marty's real IP
TEXT = "Hi. Abraham...... Hi Abraham. I am Marty, nice to meet you guys."
VOICE = "MALE"                   # FEMALE, MALE, KITTEN, ALIEN, GIANT, BOLT, STARLIGHT, WHIRLWIND
VOICE_SPEED = 1                 # 0.1 to 2
GESTURES = True                    # gesture while speaking
GESTURE_ANGLE = 30
GESTURE_TIME = 700                 # ms
# -----------------------------------------------


def _gesture_until(marty, done):
    side = 0
    while not done.is_set():
        try:
            marty.arms(*((GESTURE_ANGLE, 0) if side == 0 else (0, GESTURE_ANGLE)),
                       GESTURE_TIME)
        except Exception as e:
            print("Gesture failed:", e)
            break
        side = 1 - side
        time.sleep(GESTURE_TIME / 1000)
    try:
        marty.arms(0, 0, GESTURE_TIME)
    except Exception:
        pass


def marty_speak(marty, text, gestures=GESTURES):
    """Speak on Marty's speaker; gesture until speech finishes. Blocks."""
    done = threading.Event()

    def _speak():
        try:
            ok = marty.speak(text, blocking=True)
            if ok is False:
                print("speak() returned False: speech did not play")
        except Exception as e:
            print("Speak failed:", e)
        finally:
            done.set()

    t = threading.Thread(target=_speak, daemon=True)
    t.start()
    if gestures:
        _gesture_until(marty, done)
    t.join()


if __name__ == "__main__":
    marty = Marty("wifi", MARTY_IP)

    if not hasattr(marty, "speak"):
        raise SystemExit("Your martypy has no speak(). Run: pip install --upgrade martypy")

    marty.get_ready()
    try:
        marty.set_voice(VOICE)
        marty.set_voice_speed(VOICE_SPEED)
    except Exception as e:
        print("Could not set voice (continuing with default):", e)

    try:
        marty_speak(marty, TEXT)
    finally:
        marty.stand_straight()
        marty.close()
