"""
Marty raises an arm when the student raises a hand.

Uses run() from hand_raise.py, which calls on_raise(side) with 'left'/'right'.
Marty's movement runs in a background thread so the camera loop keeps going.

Run order for testing:
  1. python hand_raise.py          (detector alone)
  2. python marty_test_arms.py     (or the snippet in the guide: arm direction)
  3. python marty_mirror.py        (everything together)
"""
import csv
import threading
import time

from martypy import Marty
from hand_raise import run

# ---------------- Configuration ----------------
MARTY_IP = "10.142.167.12"   # <-- CHANGE: Marty's IP address (shown in the Marty app / router)
CONDITION = "reactive"      # set per session; it is written to the log

ARM_UP = 90                 # <-- TEST: if Marty's arm goes DOWN, change to -90
ARM_DOWN = 0
MOVE_TIME = 1000            # milliseconds for each arm movement
HOLD_TIME = 2.0             # seconds to keep the arm raised
MIRROR = True               # True: student's left hand -> Marty's right arm
                            # (as if looking in a mirror). False: same side.
# -----------------------------------------------

marty = Marty("wifi", MARTY_IP)
marty.get_ready()

log_file = open("log.csv", "a", newline="")
writer = csv.writer(log_file)
busy = threading.Event()    # set while Marty is moving


def log_event(event):
    writer.writerow([time.time(), CONDITION, event])
    log_file.flush()


def _raise_arm(side):
    """Raise one arm, hold, lower. Runs in a background thread."""
    try:
        marty_side = ("right" if side == "left" else "left") if MIRROR else side
        log_event(f"hand_raise_detected_{side}")

        try:
            marty.eyes("wide")          # check valid pose names in the docs
        except Exception as e:
            print("eyes failed (continuing):", e)

        left, right = (ARM_UP, ARM_DOWN) if marty_side == "left" else (ARM_DOWN, ARM_UP)
        marty.arms(left, right, MOVE_TIME)
        log_event(f"marty_arm_up_{marty_side}")

        time.sleep(HOLD_TIME)

        marty.arms(ARM_DOWN, ARM_DOWN, MOVE_TIME)
        try:
            marty.eyes("normal")
        except Exception:
            pass
        log_event("marty_arm_down")
    except Exception as e:
        print("Marty movement failed:", e)
        log_event(f"marty_error_{e}")
    finally:
        busy.clear()


def on_raise(side):
    """Called by hand_raise.run(). Returns immediately."""
    if busy.is_set():
        return                       # ignore if Marty is already moving
    busy.set()
    threading.Thread(target=_raise_arm, args=(side,), daemon=True).start()


if __name__ == "__main__":
    log_event("session_start")
    try:
        run(on_raise)
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
