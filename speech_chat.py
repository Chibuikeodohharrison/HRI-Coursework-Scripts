"""
Laptop microphone -> speech-to-text -> simple reply logic -> Marty speaks.
(Version 2: fixes 'instantly says it did not catch that' and 'cut off too soon'.)

Install:
    conda install -c conda-forge pyaudio
    python -m pip install SpeechRecognition martypy
Needs in the same folder: marty_speak_native.py
Needs internet (Google's free web speech recogniser, no API key).
"""
import csv
import time
import wave

import speech_recognition as sr
from martypy import Marty

from marty_speak_native import marty_speak

# ---------------- Configuration ----------------
MARTY_IP = "10.142.167.12"        # <-- CHANGE to Marty's real IP
CONDITION = "reactive"           # set per session; written to the log
LANGUAGE = "en-GB"
MIC_INDEX = 9                 # None = default mic. See LIST_MICS below.
LIST_MICS = False                # True = print microphones and exit

# Listening behaviour (tune these)
PUSH_TO_TALK = False             # True = press Enter, then speak (most robust for demos)
SETTLE_SECONDS = 1             # wait after Marty finishes talking before listening
PAUSE_THRESHOLD = 2            # seconds of silence that END your sentence (higher = more time)
NON_SPEAKING_DURATION = 1      # silence kept at the edges (must be <= PAUSE_THRESHOLD)
LISTEN_TIMEOUT = 5              # seconds to wait for you to START speaking
PHRASE_LIMIT = 20                # max seconds for one sentence
MIN_PHRASE_SECONDS = 1         # shorter clips are treated as noise and ignored
ENERGY_THRESHOLD = None          # None = auto-calibrate. Or set a number (e.g. 400-1500)
FAILS_BEFORE_APOLOGY = 2         # say "could not catch that" only after this many failures

GESTURES = True
SAVE_DEBUG_AUDIO = False         # True = save last_heard.wav so you can hear what the mic got
                                 # (turn OFF for real participants: privacy)
# -----------------------------------------------

log_file = open("speech_log.csv", "a", newline="", encoding="utf-8")
writer = csv.writer(log_file)


def log_event(event, text=""):
    writer.writerow([time.time(), CONDITION, event, text])
    log_file.flush()


def reply_to(text):
    """Decide what Marty says. Simple keyword rules; replace with an LLM later."""
    t = " " + text.lower() + " "
    if any(w in t for w in (" goodbye ", " bye ", " that's all ")):
        return "Goodbye. Call me if you need anything else."
    if any(w in t for w in (" hello ", " hi ", " hey ")):
        return "Hi. How may I help you today?"
    if any(w in t for w in ("stuck", "help", "hint", "don't understand", "confused")):
        return "No problem. Tell me which step you are on, and I will give you a hint."
    if any(w in t for w in ("thank", "thanks")):
        return "You are welcome."
    if any(w in t for w in ("step", "question", "equation", "problem")):
        return "Okay. Try writing down what you know first. What are you given?"
    return "Sorry, I did not catch that. Could you say it again?"


def clip_seconds(audio):
    return len(audio.frame_data) / (audio.sample_rate * audio.sample_width)


def save_wav(audio, path="last_heard.wav"):
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(audio.sample_width)
        f.setframerate(audio.sample_rate)
        f.writeframes(audio.frame_data)


def listen_once(recognizer, source):
    """Return text; '' if heard but not understood; None if silence/noise/error."""
    if PUSH_TO_TALK:
        input("Press Enter, then speak... ")
    try:
        audio = recognizer.listen(source, timeout=LISTEN_TIMEOUT,
                                  phrase_time_limit=PHRASE_LIMIT)
    except sr.WaitTimeoutError:
        return None

    secs = clip_seconds(audio)
    print(f"  (captured {secs:.1f}s of audio, energy threshold "
          f"{recognizer.energy_threshold:.0f})")
    if SAVE_DEBUG_AUDIO:
        save_wav(audio)

    if secs < MIN_PHRASE_SECONDS:
        print("  (too short, ignored as noise)")
        return None

    try:
        return recognizer.recognize_google(audio, language=LANGUAGE)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        print("Speech service error (internet?):", e)
        log_event("stt_error", str(e))
        return None


def main():
    if LIST_MICS:
        for i, name in enumerate(sr.Microphone.list_microphone_names()):
            print(i, name)
        return

    marty = Marty("wifi", MARTY_IP)
    marty.get_ready()

    recognizer = sr.Recognizer()
    recognizer.pause_threshold = PAUSE_THRESHOLD
    recognizer.non_speaking_duration = min(NON_SPEAKING_DURATION, PAUSE_THRESHOLD)
    recognizer.phrase_threshold = 0.3

    log_event("session_start")
    fails = 0
    try:
        with sr.Microphone(device_index=MIC_INDEX) as source:
            print("Calibrating for background noise (stay quiet)...")
            recognizer.adjust_for_ambient_noise(source, duration=2.0)
            if ENERGY_THRESHOLD is not None:
                recognizer.dynamic_energy_threshold = False
                recognizer.energy_threshold = ENERGY_THRESHOLD
            print(f"Ready. Energy threshold: {recognizer.energy_threshold:.0f}. "
                  "Say 'goodbye' to stop.")

            while True:
                print("Listening...")
                text = listen_once(recognizer, source)

                if text is None:
                    continue                          # silence, noise or error: just listen again

                if text == "":
                    fails += 1
                    print("(heard something, could not understand)")
                    log_event("not_understood")
                    if fails < FAILS_BEFORE_APOLOGY:
                        continue                      # stay quiet, let the person retry
                    fails = 0
                    reply = "Sorry, I did not catch that. Could you say it again?"
                else:
                    fails = 0
                    print("You said:", text)
                    log_event("heard", text)
                    reply = reply_to(text)

                print("Marty:", reply)
                log_event("reply", reply)
                marty_speak(marty, reply, gestures=GESTURES)   # blocks until finished
                time.sleep(SETTLE_SECONDS)                     # let sound and servo noise die down

                if text and any(w in text.lower() for w in ("goodbye", "bye", "that's all")):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        log_event("session_end")
        log_file.close()
        try:
            marty.stand_straight()
            marty.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()