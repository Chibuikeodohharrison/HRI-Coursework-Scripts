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
import os
import ast
import operator
import math
import json

from openai import OpenAI
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


# ---------------- Groq LLM configuration ----------------

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    timeout=15.0,
    max_retries=1,
)

MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = """
You are Marty V2, a friendly social robot interacting
with a student in a maths classroom.

Understand the meaning of the student's words instead
of relying on keywords.

Speak naturally and warmly. Remember the recent
conversation and respond to follow-up questions.

Help students learn by giving hints and asking questions
before revealing complete solutions.

Respond appropriately to greetings, jokes, opinions,
confusion and everyday conversation.

Use simple English and keep spoken responses to one
to three short sentences.

Do not use markdown or lists in spoken replies.
Never claim to have moved or performed a physical action
unless the robot controller actually did it.
"""

conversation_history = []
MAX_HISTORY_MESSAGES = 10


# ---------------- Safe maths calculator ----------------

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def calculate(expression):
    """Evaluate basic arithmetic without using eval()."""

    tree = ast.parse(expression, mode="eval")

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                if isinstance(node.value, bool):
                    raise ValueError("Boolean values are not allowed")
                return node.value
            raise ValueError("Only numbers are allowed")

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)

            if op_type not in OPERATORS:
                raise ValueError("Unsupported operator")

            left = evaluate(node.left)
            right = evaluate(node.right)

            # Avoid unnecessarily huge calculations.
            if op_type is ast.Pow:
                if abs(right) > 100 or abs(left) > 1e100:
                    raise ValueError("Exponent is too large")

            result = OPERATORS[op_type](left, right)

            if isinstance(result, (int, float)):
                if not math.isfinite(result):
                    raise ValueError("Result is not finite")

            return result

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)

            if op_type not in OPERATORS:
                raise ValueError("Unsupported operator")

            return OPERATORS[op_type](evaluate(node.operand))

        raise ValueError("Unsupported mathematical expression")

    result = evaluate(tree)

    # Limit excessively large results.
    if isinstance(result, int) and result.bit_length() > 4096:
        raise ValueError("Result is too large")

    return result


CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": (
            "Calculate a mathematical expression exactly. "
            "Use this for arithmetic questions. "
            "Pass only a valid arithmetic expression such as "
            "'245 * 38' or '(25 + 15) / 2'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The arithmetic expression to calculate"
                }
            },
            "required": ["expression"],
            "additionalProperties": False
        }
    }
}


def reply_to(text):
    """Chat naturally and use Python for accurate arithmetic."""
    global conversation_history

    if not text or not text.strip():
        return "Could you say that again, please?"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *conversation_history,
        {"role": "user", "content": text.strip()},
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=[CALCULATOR_TOOL],
            tool_choice="auto",
            max_tokens=180,
        )

        # Allow the LLM to request a calculation.
        for _ in range(3):
            message = response.choices[0].message

            if not message.tool_calls:
                reply = message.content or (
                    "Could you explain that another way?"
                )
                break

            messages.append(message.model_dump(exclude_none=True))

            for tool_call in message.tool_calls:
                if tool_call.function.name != "calculate":
                    continue

                try:
                    arguments = json.loads(
                        tool_call.function.arguments
                    )
                    result = calculate(arguments["expression"])

                    tool_result = {
                        "success": True,
                        "result": str(result),
                    }

                except Exception:
                    tool_result = {
                        "success": False,
                        "error": (
                            "That calculation could not be evaluated. "
                            "Please try a simpler expression."
                        ),
                    }

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result),
                })

            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=[CALCULATOR_TOOL],
                tool_choice="auto",
                max_tokens=180,
            )

        else:
            reply = (
                "I couldn't finish that calculation. "
                "Could we try it another way?"
            )

        reply = reply.strip()

        conversation_history.extend([
            {"role": "user", "content": text.strip()},
            {"role": "assistant", "content": reply},
        ])

        conversation_history = conversation_history[
            -MAX_HISTORY_MESSAGES:
        ]

        return reply

    except Exception as error:
        print("Groq error:", error)
        log_event("llm_error", str(error))
        return "I'm having trouble thinking right now. Please try again."


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