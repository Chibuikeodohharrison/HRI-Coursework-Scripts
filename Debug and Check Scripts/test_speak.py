"""
Diagnose why speech fails. Runs three checks in order and prints PASS/FAIL.
Run:  python test_speak.py
"""
import traceback
import urllib.request

from martypy import Marty

MARTY_IP = "10.53.109.234"      # <-- CHANGE to Marty's real IP
TEXT = "Hello there"

# 1. Does the laptop have working internet?
print("1) Internet check...")
try:
    status = urllib.request.urlopen("https://www.google.com", timeout=5).status
    print("   PASS (HTTP", status, ")")
except Exception as e:
    print("   FAIL:", e)

# 2. Connect to Marty
print("2) Connecting to Marty...")
try:
    m = Marty("wifi", MARTY_IP)
    print("   PASS")
except Exception:
    traceback.print_exc()
    raise SystemExit("Cannot connect to Marty; fix the IP/network first.")

# 3. Speak on the computer speaker (uses the same online speech service)
print("3) speak_on_computer ...")
try:
    m.speak_on_computer(TEXT)
    print("   PASS (you should have heard it from the laptop)")
except Exception:
    print("   FAIL:")
    traceback.print_exc()

# 4. Speak on Marty's speaker
print("4) speak on Marty's speaker ...")
try:
    m.speak(TEXT, blocking=True)
    print("   PASS (you should have heard it from Marty)")
except Exception:
    print("   FAIL:")
    traceback.print_exc()

m.close()
