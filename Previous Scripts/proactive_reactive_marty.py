"""
PROACTIVE CONDITION: Marty interprets the context and offers help FIRST.

  One YOLO pass per frame -> per-person detectors:
     * hand raise   -> help REQUEST   (same as the reactive condition)
     * inactivity   -> "stuck" signal
     * confusion-like expression -> "stuck" signal
  A "stuck" signal (see TRIGGER_RULE) makes Marty orient/gesture and OFFER help.
  If the student accepts, the SAME dialogue as the reactive condition runs
  (reply_to + speak_reply from speech4.py).

The ONLY difference from the reactive condition is who initiates. Everything after help starts
is shared code, so the two conditions can be compared fairly.

TRIGGER_RULE:
  "either" - inactivity OR confusion triggers an offer (more offers, more false alarms)
  "both"   - needs both within COMBINE_WINDOW seconds (fewer, better-justified offers)

Needs in the same folder: perception_yolo.py, detect_inactivity_yolo.py,
confusion_detect_yolo.py, speech4.py, speech_utils2.py
"""
import queue
import threading
import time

import cv2
import speech_recognition as sr
from martypy import Marty

import perception_yolo as pv
import detect_inactivity_yolo as inact
import confusion_detect_yolo as conf
import speech4 as s4                 # LLM dialogue (reply_to), speaking, mic settings, logging

# ---------------- Configuration ----------------
CONDITION = "proactive"
s4.CONDITION = CONDITION             # every log row records the condition (overrides reactive's)

TRIGGER_RULE = "either"              # "either" or "both"
COMBINE_WINDOW = 15                  # seconds (only used when TRIGGER_RULE == "both")
OFFER_COOLDOWN = 60                  # seconds before the SAME person can be offered help again
OFFER_LINE = "Hi, person {n}. You look a bit stuck. Would you like some help?"
DECLINE_REPLY = "No problem. Raise your hand if you need me."
OFFER_LISTEN_TIMEOUT = 8             # seconds to wait for an answer to the offer
DECLINE_WORDS = ("no", "nope", "no thanks", "no thank you", "i'm fine", "i am fine",
                 "i'm good", "i am good", "not now", "leave me", "go away")
OFFER_GESTURE_ANGLE = 60             # smaller than a raised hand so it reads as an "offer"

# Shared with the reactive condition (keep identical in both!)
ACK_LINE = "Yes, person {n}. How can I help?"   # keep short (speech cuts off past ~18 words)
RETRY_LINE = "Sorry, could you say that again?"
FIRST_LISTEN_TIMEOUT = 10            # seconds to wait for the student to start speaking
FOLLOWUP_TIMEOUT = 8                 # seconds to wait for a follow-up question
MAX_TURNS = 5                        # max exchanges per session
END_WORDS = ("goodbye", "bye", "that's all")
ACK_ARM_ANGLE = 90                   # flip the sign if Marty's arm goes DOWN
ACK_MOVE_MS = 700
MIRROR = True                        # student's left hand -> Marty's right arm
# -----------------------------------------------

events = queue.Queue()               # ("request"|"offer", person_number, detail, t)
busy_people = set()                  # people queued or being served
last_session = {}                    # person number -> time their last session ended
recent = {}                          # (person number, "inactive"/"confused") -> time
lock = threading.Lock()
ready = threading.Event()
stop = threading.Event()


# ---------------- Marty actions and listening (same as the reactive condition) ----------------
def arm_up(marty, side):
    """Raise an arm (both for 'both'). arms() returns immediately, so Marty moves
    while the speech is being prepared."""
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


def listen(recognizer, source, timeout):
    """Return text; '' if heard but not understood; None if silence/noise/error."""
    try:
        audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=s4.PHRASE_LIMIT)
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


# ---------------- Camera-side: signals -> events ----------------
def submit(kind, person, detail):
    """Queue an event unless that person is already waiting/being served or on cooldown."""
    if not ready.is_set():
        return
    now = time.time()
    with lock:
        if person in busy_people:
            return
        if kind == "offer" and now - last_session.get(person, float("-inf")) < OFFER_COOLDOWN:
            return
        busy_people.add(person)
    s4.log_event(f"{kind}_trigger_p{person}", detail)
    print(f"[camera] {kind} for person {person}: {detail}")
    events.put((kind, person, detail, now))


def on_signal(person, source):
    """A 'stuck' signal arrived ('inactive' or 'confused')."""
    now = time.time()
    s4.log_event(f"signal_{source}_p{person}", TRIGGER_RULE)        # logged under both rules
    recent[(person, source)] = now
    if TRIGGER_RULE == "either":
        submit("offer", person, source)
        return
    other = "confused" if source == "inactive" else "inactive"
    if now - recent.get((person, other), float("-inf")) <= COMBINE_WINDOW:
        submit("offer", person, "inactive+confused")


# ---------------- Conversation (shared with reactive) ----------------
def dialogue_loop(marty, recognizer, source, person, first_text=None, first_timeout=None):
    """Same turn-taking as the reactive condition. If first_text is given
    (the student already answered the offer), it is processed as the first turn."""
    turns, timeout, retried = 0, first_timeout or FIRST_LISTEN_TIMEOUT, False
    end_reason, text = "max_turns", first_text
    while turns < MAX_TURNS:
        if text is None:
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
            text = None
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
        text = None
    return end_reason


def handle_request(marty, recognizer, source, person, side, t0):
    """Student raised a hand: identical behaviour to the reactive condition."""
    s4.log_event(f"session_start_p{person}", f"request:{side}")
    arm_up(marty, side)
    say(marty, ACK_LINE.format(n=person))
    s4.log_event("ack_speak_done", f"{time.time() - t0:.2f}s after raise")
    arm_down(marty)
    time.sleep(s4.SETTLE_SECONDS)
    reason = dialogue_loop(marty, recognizer, source, person)
    s4.log_event(f"session_end_p{person}", reason)


def handle_offer(marty, recognizer, source, person, why, t0):
    """Marty speaks first. Student may accept, decline or ignore."""
    s4.log_event(f"session_start_p{person}", f"offer:{why}")
    # a smaller 'offering' gesture on the side facing the student, then the offer line
    try:
        marty.arms(OFFER_GESTURE_ANGLE, 0, ACK_MOVE_MS)
    except Exception as e:
        print("Gesture failed:", e)
    say(marty, OFFER_LINE.format(n=person))
    s4.log_event("offer_spoken", f"{time.time() - t0:.2f}s after trigger")
    arm_down(marty)
    time.sleep(s4.SETTLE_SECONDS)

    answer = listen(recognizer, source, OFFER_LISTEN_TIMEOUT)
    if answer is None:
        s4.log_event(f"offer_ignored_p{person}")
        reason = "offer_ignored"
    elif answer == "":
        s4.log_event(f"offer_unclear_p{person}")
        say(marty, RETRY_LINE)
        time.sleep(s4.SETTLE_SECONDS)
        answer = listen(recognizer, source, OFFER_LISTEN_TIMEOUT)
        if answer:
            reason = _after_answer(marty, recognizer, source, person, answer)
        else:
            reason = "offer_unclear"
    else:
        reason = _after_answer(marty, recognizer, source, person, answer)
    s4.log_event(f"session_end_p{person}", reason)
    return reason


def _after_answer(marty, recognizer, source, person, answer):
    low = " " + answer.lower().strip(" .!") + " "
    declined = any(low.strip() == w or low.startswith(f" {w} ") for w in DECLINE_WORDS)
    if declined:
        s4.log_event(f"offer_declined_p{person}", answer)
        say(marty, DECLINE_REPLY)
        return "offer_declined"
    s4.log_event(f"offer_accepted_p{person}", answer)
    return "offer_accepted:" + dialogue_loop(marty, recognizer, source, person, first_text=answer)


# ---------------- Worker thread (all Marty + microphone work) ----------------
def worker(marty):
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = s4.PAUSE_THRESHOLD
    recognizer.non_speaking_duration = min(s4.NON_SPEAKING_DURATION, s4.PAUSE_THRESHOLD)
    recognizer.phrase_threshold = 0.3

    with sr.Microphone(device_index=s4.MIC_INDEX) as source:
        print("Calibrating microphone (stay quiet)...")
        recognizer.adjust_for_ambient_noise(source, duration=2.0)
        if s4.ENERGY_THRESHOLD is not None:
            recognizer.dynamic_energy_threshold = False
            recognizer.energy_threshold = s4.ENERGY_THRESHOLD
        print(f"Ready (proactive, rule={TRIGGER_RULE}). Marty will offer help if someone looks stuck; "
              "students can also raise a hand.")
        ready.set()

        while not stop.is_set():
            try:
                kind, person, detail, t0 = events.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if kind == "request":
                    handle_request(marty, recognizer, source, person, detail, t0)
                else:
                    handle_offer(marty, recognizer, source, person, detail, t0)
            except Exception as e:
                print("Session error:", e)
                s4.log_event("session_error", str(e))
            finally:
                with lock:
                    busy_people.discard(person)
                    last_session[person] = time.time()


# ---------------- Main: one camera loop, one YOLO pass per frame ----------------
def main():
    marty = Marty("wifi", s4.MARTY_IP)
    marty.get_ready()
    s4.log_event("run_start", f"{CONDITION}:{TRIGGER_RULE}")
    thread = threading.Thread(target=worker, args=(marty,), daemon=True)
    thread.start()

    tracker = pv.PeopleTracker()
    hands = pv.HandRaiseDetector()
    inactivity = inact.InactivityDetector()
    confusion = conf.ConfusionDetector()
    cap = pv.open_camera()

    try:
        while cap is not None and cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("Failed to read camera frame.")
                break
            now = time.monotonic()
            people = tracker.process(frame)

            raise_events, highlight = hands.update(people, now)
            for p, side in raise_events:
                submit("request", p.number, side)
            idle_events, idle = inactivity.update(people, now)
            for p, _secs in idle_events:
                on_signal(p.number, "inactive")
            conf_events, face_status = confusion.update(frame, people, now)
            for p in conf_events:
                on_signal(p.number, "confused")

            notes = {p.track_id: [f"idle {idle.get(p.track_id, 0):.0f}s",
                                  f"{face_status.get(p.track_id, ('', 0))[0]}"] for p in people}
            pv.draw_people(frame, people, notes=notes, highlight=highlight)
            cv2.putText(frame, f"PROACTIVE ({TRIGGER_RULE})   people: {len(people)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(frame, f"PROACTIVE ({TRIGGER_RULE})   people: {len(people)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Proactive Marty - Q quit, R recalibrate faces", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                confusion.recalibrate()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
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
