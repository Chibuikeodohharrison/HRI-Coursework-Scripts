"""Test Marty's connection and arm direction on their own, before using the camera."""
import time
from martypy import Marty

MARTY_IP = "10.12.104.234"   # <-- CHANGE

marty = Marty("wifi", MARTY_IP)
marty.get_ready()

print("Right arm to +90 ... does it go UP or DOWN?")
marty.arms(0, 90, 1000)      # (left_angle, right_angle, move_time_ms)
time.sleep(2.5)

print("Left arm to +90 ...")
marty.arms(90, 0, 1000)
time.sleep(2.5)

print("Both to -90 ...")
marty.arms(-90, -90, 1000)
time.sleep(2.5)

marty.arms(0, 0, 1000)
time.sleep(1.5)
marty.stand_straight()
marty.close()
print("Done. Note which sign raises the arm and set ARM_UP in marty_mirror.py.")
