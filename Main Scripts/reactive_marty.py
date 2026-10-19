"""
REACTIVE CONDITION: the student does something (raises a hand) -> Marty responds.

  camera (YOLO, up to 5 people)  --hand raised-->  queue  -->  conversation worker
  worker: Marty raises an arm + says "Yes, person N. How can I help?"
          -> listens (laptop mic) -> Groq LLM reply (speech4.reply_to) -> Marty speaks
          -> follow-up turns until goodbye / silence / MAX_TURNS -> back to idle

Design notes
  * The camera loop runs in the MAIN thread (OpenCV window); all Marty and microphone
    work runs in ONE worker thread, so martypy is never called from two threads.
  * Hand raises are queued in order. A person who is already waiting is not queued twice.
  * The dialogue (reply_to, speak_reply, listening settings) is reused from speech4.py,
    so the proactive condition can later share exactly the same conversation code.
  * Every step is logged (speech_log.csv), including the delay from hand raise to
    Marty speaking, which you can report as a measure.

Needs in the same folder: speech4.py, speech_utils2.py, hand_raise_yolo.py
"""
import queue
import threading
import time

import speech_recognition as sr
from martypy import Marty

import speech4 as s4                 # LLM dialogue, speaking, mic settings, logging
import hand_raise_yolo2 as camera     # multi-person hand-raise detector

# ---------------- Configuration ----------------
MARTY_IP = s4.MARTY_IP
CONDITION = "reactive"
s4.CONDITION = CONDITION             # so every log row records the condition

ACK_LINE = "Yes, person {n}. How can I help?"   # keep it short (speech cuts off past ~18 words)
RETRY_LINE = "Sorry, could you say that again?"
FIRST_LISTEN_TIMEOUT = 10            # seconds to wait for the student to start speaking
FOLLOWUP_TIMEOUT = 8                 # seconds to wait for a follow-up question
MAX_TURNS = 10                        # max exchanges per hand raise
END_WORDS = ("goodbye", "bye", "that's all")

ACK_ARM_ANGLE = 90                   # flip the sign if Marty's arm goes DOWN
ACK_MOVE_MS = 700
MIRROR = True                        # student's left hand -> Marty's right arm
# -----------------------------------------------

pending = queue.Queue()              # (person, side, time_of_raise)
queued = set()                       # people currently waiting or being served
lock = threading.Lock()
ready = threading.Event()            # set once the microphone is calibrated
stop = threading.Event()


# ---------------- Camera callback ----------------
def on_raise(person, side):
    """Called by the camera loop. Must return quickly."""
    if not ready.is_set():
        return                       # mic not calibrated yet
    with lock:
        if person in queued:
            return                   # already waiting or being served
        queued.add(person)
    t = time.time()
    print(f"[camera] person {person} raised {side} hand")
    s4.log_event(f"hand_raise_p{person}", side)
    pending.put((person, side, t))


# ---------------- Marty actions ----------------
def arm_up(marty, side):
    """Raise an arm (both for 'both'). arms() returns immediately, so Marty
    moves while the speech is being prepared (no extra delay)."""
    a = ACK_ARM_ANGLE
    if side == "both":
        left, right = a, a
    else:
        marty_side = ("right" if side == "left" else "left") if MIRROR else side
        left, right = (a, 0) if marty_side == "left" else (0, a)
    try:
        marty.arms(left, right, ACK_MOVE_MS)
    except Exception as e:
        print("Arm raise failed:", e)


def arm_down(marty):
    try:
        marty.arms(0, 0, ACK_MOVE_MS)
    except Exception as e:
        print("Arm lower failed:", e)


def say(marty, text, gestures=False):
    s4.speak_reply(marty, text, gestures=gestures, log_fn=s4.log_event)


# ---------------- Listening ----------------
def listen(recognizer, source, timeout):
    """Return text; '' if heard but not understood; None if silence/noise/error."""
    try:
        audio = recognizer.listen(source, timeout=timeout,
                                  phrase_time_limit=s4.PHRASE_LIMIT)
    except sr.WaitTimeoutError:
        return None
    if s4.clip_seconds(audio) < s4.MIN_PHRASE_SECONDS:
        return None
    try:
        return recognizer.recognize_google(audio, language=s4.LANGUAGE)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        print("Speech service error (internet?):", e)
        s4.log_event("stt_error", str(e))
        return None


# ---------------- One hand-raise conversation ----------------
def handle_hand_raise(marty, recognizer, source, person, side, t_raise):
    s4.log_event(f"session_start_p{person}", side)

    arm_up(marty, side)                                  # moves while speech is prepared
    s4.log_event("ack_speak_called", f"{time.time() - t_raise:.2f}s after raise")
    say(marty, ACK_LINE.format(n=person))
    s4.log_event("ack_speak_done", f"{time.time() - t_raise:.2f}s after raise")
    print(f"[timing] hand raise -> Marty finished acknowledging: "
          f"{time.time() - t_raise:.1f}s")
    arm_down(marty)
    time.sleep(s4.SETTLE_SECONDS)

    turns, timeout, retried = 0, FIRST_LISTEN_TIMEOUT, False
    end_reason = "max_turns"
    while turns < MAX_TURNS:
        print(f"Listening to person {person}...")
        text = listen(recognizer, source, timeout)

        if text is None:
            end_reason = "silence"
            break
        if text == "":
            s4.log_event("not_understood")
            if retried:
                end_reason = "not_understood"
                break
            retried = True
            say(marty, RETRY_LINE)
            time.sleep(s4.SETTLE_SECONDS)
            continue

        retried = False
        print(f"Person {person} said: {text}")
        s4.log_event(f"heard_p{person}", text)

        reply = s4.reply_to(text)
        print("Marty:", reply)
        s4.log_event("reply", reply)
        say(marty, reply, gestures=s4.GESTURES)
        time.sleep(s4.SETTLE_SECONDS)

        turns += 1
        timeout = FOLLOWUP_TIMEOUT
        if any(w in text.lower() for w in END_WORDS):
            end_reason = "goodbye"
            break

    s4.log_event(f"session_end_p{person}", end_reason)
    print(f"[session] person {person} finished ({end_reason}). Back to idle.")


# ---------------- Worker thread (all Marty + microphone work) ----------------
def worker(marty):
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = s4.PAUSE_THRESHOLD
    recognizer.non_speaking_duration = min(s4.NON_SPEAKING_DURATION, s4.PAUSE_THRESHOLD)
    recognizer.phrase_threshold = 0.3

    with sr.Microphone(device_index=s4.MIC_INDEX) as source:
        print("Calibrating microphone for background noise (stay quiet)...")
        recognizer.adjust_for_ambient_noise(source, duration=2.0)
        if s4.ENERGY_THRESHOLD is not None:
            recognizer.dynamic_energy_threshold = False
            recognizer.energy_threshold = s4.ENERGY_THRESHOLD
        print(f"Ready. Energy threshold: {recognizer.energy_threshold:.0f}. "
              "Raise a hand to talk to Marty.")
        ready.set()

        while not stop.is_set():
            try:
                person, side, t_raise = pending.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                handle_hand_raise(marty, recognizer, source, person, side, t_raise)
            except Exception as e:
                print("Session error:", e)
                s4.log_event("session_error", str(e))
            finally:
                with lock:
                    queued.discard(person)


def main():
    marty = Marty("wifi", MARTY_IP)
    marty.get_ready()

    s4.log_event("run_start", CONDITION)
    thread = threading.Thread(target=worker, args=(marty,), daemon=True)
    thread.start()

    try:
        camera.run(on_raise)             # blocks in the main thread; press Q to quit
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        thread.join(timeout=20)
        s4.log_event("run_end", CONDITION)
        s4.log_file.close()
        try:
            marty.stand_straight()
            marty.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()