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

from martypy import Marty
from speech_utils2 import speak_reply
import speech_recognition as sr
from openai import OpenAI
import sympy as sp

# ---------------- Configuration ----------------
MARTY_IP = "10.142.167.12"        # <-- CHANGE to Marty's real IP
CONDITION = "reactive"           # set per session; written to the log
LANGUAGE = "en-GB"
MIC_INDEX = 1                 # None = default mic. See LIST_MICS below.
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

For mathematical questions, use the calculate tool to obtain
accurate results instead of guessing or relying on memory.

Convert natural language into the tool's supported syntax.
Use * for multiplication and ** for powers.

Use solve for equations, simplify for algebraic simplification,
differentiate for derivatives, integrate for integrals, and
limit for limits.

After receiving the tool result, explain the method briefly
in simple spoken English. State the mathematical result
accurately. Do not invent intermediate steps or claim that
a symbolic expression has been evaluated when it has not.

For indefinite integration, mention the constant of integration
C when explaining the answer.
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
    """
    Safely evaluate arithmetic and common symbolic mathematics with SymPy.

    Supported commands:
      solve: 2*x + 3 = 11
      differentiate: x**2 + 3*x, x
      integrate: 2*x, x
      limit: sin(x)/x, x, 0
      simplify: (x**2 - 4)/(x - 2)
      factor: x**2 - 9
      expand: (x + 2)**2

    A plain arithmetic expression is also accepted.
    """

    if not isinstance(expression, str):
        raise ValueError("The expression must be text.")

    expression = expression.strip()

    if not expression or len(expression) > 500:
        raise ValueError("The expression is empty or too long.")

    # Allowed symbolic variables and constants.
    symbols = {
        name: sp.Symbol(name)
        for name in ("x", "y", "z", "t", "n")
    }

    constants = {
        "pi": sp.pi,
        "E": sp.E,
    }

    functions = {
        "sqrt": sp.sqrt,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,
        "log": sp.log,
        "ln": sp.log,
        "exp": sp.exp,
        "abs": sp.Abs,
    }

    def parse_expression(source):
        """Convert a restricted Python expression into a SymPy object."""

        tree = ast.parse(source.strip(), mode="eval")

        def visit(node):
            if isinstance(node, ast.Expression):
                return visit(node.body)

            if isinstance(node, ast.Constant):
                if isinstance(node.value, bool):
                    raise ValueError("Boolean values are not allowed.")

                if isinstance(node.value, int):
                    return sp.Integer(node.value)

                if isinstance(node.value, float):
                    return sp.Float(node.value)

                raise ValueError("Only numeric constants are allowed.")

            if isinstance(node, ast.Name):
                if node.id in symbols:
                    return symbols[node.id]

                if node.id in constants:
                    return constants[node.id]

                raise ValueError(
                    f"Unsupported name: {node.id}"
                )

            if isinstance(node, ast.BinOp):
                left = visit(node.left)
                right = visit(node.right)

                if isinstance(node.op, ast.Add):
                    return left + right

                if isinstance(node.op, ast.Sub):
                    return left - right

                if isinstance(node.op, ast.Mult):
                    return left * right

                if isinstance(node.op, ast.Div):
                    return left / right

                if isinstance(node.op, ast.Pow):
                    if right.is_number and right.is_real:
                        if abs(float(right)) > 100:
                            raise ValueError(
                                "Exponent is too large."
                            )
                    return left ** right

                if isinstance(node.op, ast.FloorDiv):
                    return sp.floor(left / right)

                if isinstance(node.op, ast.Mod):
                    return sp.Mod(left, right)

                raise ValueError("Unsupported operator.")

            if isinstance(node, ast.UnaryOp):
                value = visit(node.operand)

                if isinstance(node.op, ast.USub):
                    return -value

                if isinstance(node.op, ast.UAdd):
                    return value

                raise ValueError("Unsupported unary operator.")

            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise ValueError("Unsupported function call.")

                function_name = node.func.id

                if function_name not in functions:
                    raise ValueError(
                        f"Unsupported function: {function_name}"
                    )

                if node.keywords:
                    raise ValueError(
                        "Named function arguments are not supported."
                    )

                args = [visit(arg) for arg in node.args]
                return functions[function_name](*args)

            raise ValueError(
                "Unsupported mathematical expression."
            )

        return visit(tree)

    def split_arguments(source):
        """Split commas outside parentheses."""
        parts = []
        depth = 0
        start = 0

        for index, char in enumerate(source):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1

                if depth < 0:
                    raise ValueError("Mismatched parentheses.")

            elif char == "," and depth == 0:
                parts.append(source[start:index].strip())
                start = index + 1

        if depth != 0:
            raise ValueError("Mismatched parentheses.")

        parts.append(source[start:].strip())
        return parts

    def get_variable(source):
        """Return an allowed symbolic variable."""
        name = source.strip()

        if name not in symbols:
            raise ValueError(
                "Use one of these variables: x, y, z, t or n."
            )

        return symbols[name]

    def format_result(result):
        """Return a concise, readable mathematical result."""
        if isinstance(result, list):
            if not result:
                return "No solutions were found."

            return "[" + "; ".join(
                format_result(item) for item in result
            ) + "]"

        if isinstance(result, dict):
            return "; ".join(
                f"{key} = {format_result(value)}"
                for key, value in result.items()
            )

        return sp.sstr(result)

    # Determine which mathematical operation was requested.
    if ":" in expression:
        command, body = expression.split(":", 1)
        command = command.strip().lower()
        body = body.strip()
    else:
        command = "calculate"
        body = expression

    if command in ("calculate", "arithmetic"):
        # Allow ordinary arithmetic or symbolic simplification.
        if "=" in body:
            raise ValueError(
                "For equations, use 'solve: left_side = right_side'."
            )

        result = parse_expression(body)

    elif command == "simplify":
        result = sp.simplify(parse_expression(body))

    elif command == "factor":
        result = sp.factor(parse_expression(body))

    elif command == "expand":
        result = sp.expand(parse_expression(body))

    elif command == "solve":
        # Example: solve: 2*x + 3 = 11
        if body.count("=") != 1:
            raise ValueError(
                "Write an equation such as solve: 2*x + 3 = 11."
            )

        left_text, right_text = body.split("=", 1)
        left = parse_expression(left_text)
        right = parse_expression(right_text)

        equation = sp.Eq(left, right)
        variables = sorted(
            equation.free_symbols,
            key=lambda symbol: symbol.name
        )

        if not variables:
            return "True" if sp.simplify(left - right) == 0 else "False"

        result = sp.solve(equation, variables)

    elif command in ("differentiate", "derivative", "diff"):
        # Example: differentiate: x**2 + 3*x, x
        args = split_arguments(body)

        if len(args) not in (1, 2):
            raise ValueError(
                "Use differentiate: expression, variable."
            )

        expr = parse_expression(args[0])
        variable = get_variable(args[1]) if len(args) == 2 else symbols["x"]
        result = sp.diff(expr, variable)

    elif command in ("integrate", "integral", "int"):
        # Example: integrate: 2*x, x
        args = split_arguments(body)

        if len(args) not in (1, 2):
            raise ValueError(
                "Use integrate: expression, variable."
            )

        expr = parse_expression(args[0])
        variable = get_variable(args[1]) if len(args) == 2 else symbols["x"]
        result = sp.integrate(expr, variable)

    elif command == "limit":
        # Example: limit: sin(x)/x, x, 0
        args = split_arguments(body)

        if len(args) != 3:
            raise ValueError(
                "Use limit: expression, variable, point."
            )

        expr = parse_expression(args[0])
        variable = get_variable(args[1])
        point = parse_expression(args[2])
        result = sp.limit(expr, variable, point)

    else:
        raise ValueError(
            "Unsupported command. Try solve, simplify, factor, "
            "expand, differentiate, integrate, limit or calculate."
        )

    # Prevent accidental creation of excessively large outputs.
    answer = format_result(result)

    if len(answer) > 2000:
        raise ValueError("The result is too large to return.")

    return answer


CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": (
            "Use this tool whenever a maths question requires a "
            "calculation, equation solution, algebraic simplification, "
            "differentiation, integration, or a limit. Do not guess "
            "mathematical results. Convert the student's request into "
            "one supported command and a valid expression. Use * for "
            "multiplication and ** for powers. Available commands: "
            "'calculate: 245 * 38', "
            "'solve: 2*x + 3 = 11', "
            "'simplify: (x**2 - 4)/(x - 2)', "
            "'factor: x**2 - 9', "
            "'expand: (x + 2)**2', "
            "'differentiate: x**2 + 3*x, x', "
            "'integrate: 2*x, x', "
            "'limit: sin(x)/x, x, 0'. "
            "Supported variables are x, y, z, t and n. "
            "Return the tool result to the student in simple English."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "A supported mathematical command and expression."
                    )
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

                    print(
                        f"[SYMPY TOOL] Input: {arguments['expression']} "
                        f"| Result: {result}"
                    )

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

                print("Marty is speaking...")
                speak_reply(marty, reply, gestures=GESTURES, log_fn=log_event)
                time.sleep(SETTLE_SECONDS)     # let the room go quiet before listening
                print("Ready for the next question.")

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