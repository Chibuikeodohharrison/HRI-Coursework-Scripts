"""
Find out why Marty's speech cuts off. Each test ends with the word "goodbye",
so you can tell by ear whether the sentence was cut before the end.

Run:  python speak_cutoff_test.py
Keep marty_speak_native.py in the same folder.
After each test, type y if you heard the final word "goodbye", n if it was cut off.
"""
import time

from martypy import Marty
from marty_speak_native import marty_speak

MARTY_IP = "10.142.167.12"     # <-- your Marty's IP

SHORT = "Hello there. Nice to meet you, goodbye."
MEDIUM = ("Let us look at this equation together. First, move the three "
          "to the other side, and then goodbye.")
LONG = ("Okay, so to solve this equation we first subtract three from both sides, "
        "then we divide both sides by two, and finally we check our answer by "
        "putting it back into the original equation, so that is all, goodbye.")

results = []


def ask(label, text, elapsed):
    words = len(text.split())
    print(f"   speak() took {elapsed:.1f}s for {words} words "
          f"({words / max(elapsed, 0.1):.1f} words/s)")
    heard = input("   Did you hear the final word 'goodbye'? (y/n): ").strip().lower()
    results.append((label, words, round(elapsed, 1), heard))


m = Marty("wifi", MARTY_IP)
m.get_ready()
try:
    m.set_voice("MALE")
except Exception:
    pass

tests = [
    ("A: plain speak, SHORT, no gestures", SHORT, False),
    ("B: plain speak, MEDIUM, no gestures", MEDIUM, False),
    ("C: plain speak, LONG, no gestures", LONG, False),
    ("D: with gestures, SHORT", SHORT, True),
    ("E: with gestures, LONG", LONG, True),
]

for label, text, gestures in tests:
    print(f"\n{label}")
    time.sleep(2)
    start = time.time()
    if gestures:
        marty_speak(m, text, gestures=True)
    else:
        m.speak(text, blocking=True)
    ask(label, text, time.time() - start)

print("\n===== SUMMARY =====")
for r in results:
    print(r)

m.stand_straight()
m.close()