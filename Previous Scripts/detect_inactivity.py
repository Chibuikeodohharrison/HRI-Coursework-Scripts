"""
Inactivity detectors. Both call on_inactive() once when no activity has been
seen for INACTIVITY_SECONDS, then stay quiet until activity resumes AND the
cooldown has passed. Thresholds are heuristic and should be tuned in pilots.
"""
import time
import threading
import cv2
from pynput import keyboard, mouse

INACTIVITY_SECONDS = 10
COOLDOWN_SECONDS = 20       # minimum gap between triggers
CHECK_INTERVAL = 0.2


# ---------- Version A: keyboard / mouse inactivity ----------
# class InputInactivityDetector:
#     def __init__(self, on_inactive, seconds=INACTIVITY_SECONDS):
#         self.on_inactive = on_inactive
#         self.seconds = seconds
#         self.last_activity = time.time()
#         self.last_trigger = 0
#         self.fired = False          # fired for the current idle period?
#         self._stop = threading.Event()

#     def _activity(self, *args):
#         self.last_activity = time.time()
#         self.fired = False          # activity resumed: re-arm

#     def run(self):
#         kl = keyboard.Listener(on_press=self._activity)
#         ml = mouse.Listener(on_move=self._activity, on_click=self._activity,
#                             on_scroll=self._activity)
#         kl.start(); ml.start()
#         try:
#             while not self._stop.is_set():
#                 now = time.time()
#                 idle = now - self.last_activity
#                 if (idle >= self.seconds and not self.fired
#                         and now - self.last_trigger > COOLDOWN_SECONDS):
#                     self.fired = True
#                     self.last_trigger = now
#                     print(f">>> INACTIVE for {idle:.1f}s -> trigger")
#                     self.on_inactive()
#                 time.sleep(CHECK_INTERVAL)
#         finally:
#             kl.stop(); ml.stop()

#     def stop(self):
#         self._stop.set()


# ---------- Version B: webcam motion inactivity ----------
MOTION_THRESHOLD = 2.0      # mean pixel difference counted as "movement"; tune this
DOWNSCALE = (320, 240)

class MotionInactivityDetector:
    def __init__(self, on_inactive, seconds=INACTIVITY_SECONDS,
                 camera_index=0, show=True):
        self.on_inactive = on_inactive
        self.seconds = seconds
        self.camera_index = camera_index
        self.show = show

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        prev = None
        last_activity = time.time()
        last_trigger, fired = 0, False

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(cv2.resize(frame, DOWNSCALE), cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)   # suppress sensor noise
            now = time.time()
            motion = 0.0
            if prev is not None:
                motion = float(cv2.absdiff(gray, prev).mean())
                if motion > MOTION_THRESHOLD:
                    last_activity = now
                    fired = False
            prev = gray

            idle = now - last_activity
            if idle >= self.seconds and not fired and now - last_trigger > COOLDOWN_SECONDS:
                fired, last_trigger = True, now
                print(f">>> NO MOVEMENT for {idle:.1f}s -> trigger")
                self.on_inactive()

            if self.show:
                cv2.putText(frame, f"motion {motion:.1f}  idle {idle:.0f}s",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Motion inactivity (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Test version A: stop touching keyboard/mouse for 10 s (Ctrl+C to quit)
    # InputInactivityDetector(lambda: print("(test) student inactive")).run()
    # Or test version B instead:
    MotionInactivityDetector(lambda: print("(test) student inactive")).run()