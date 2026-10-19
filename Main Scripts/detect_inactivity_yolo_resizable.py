"""
Per-person INACTIVITY detector using YOLO pose (replaces the whole-frame motion version).

For every tracked person it measures how far their upper-body keypoints (nose, shoulders,
elbows, wrists) have moved from an "anchor" pose, in units of that person's shoulder width,
so the result does not depend on how far they sit from the camera. If they stay within
MOVEMENT_THRESHOLD of the anchor for INACTIVITY_SECONDS, that person is "inactive".

  on_inactive(person_number, idle_seconds)   fires once per idle period, per person.

Writing, turning a page, or leaning counts as activity; sitting still does not.
10 s is short for maths (silent thinking is normal): keep it for the demo and justify a
longer value from pilot data in the real study.

Run on its own to tune:  python detect_inactivity_yolo.py
Needs: perception_yolo.py in the same folder.
"""
import math
import time

import cv2

import perception_yolo_resizable as pv

# ---------------- Configuration ----------------
INACTIVITY_SECONDS = 10
COOLDOWN_SECONDS = 20            # per person, between triggers
MOVEMENT_THRESHOLD = 0.12        # mean joint movement (in shoulder widths) that counts as activity
JOINTS = [0, 5, 6, 7, 8, 9, 10]  # nose, shoulders, elbows, wrists
MIN_COMMON_JOINTS = 3            # need this many joints visible to compare
FORGET_AFTER_SECONDS = 10        # drop state for people not seen for this long
WINDOW_DIMENSIONS = (640, 360)

# -----------------------------------------------


class _State:
    def __init__(self, now):
        self.anchor = {}             # joint -> (x, y)
        self.last_activity = now
        self.fired = False
        self.last_trigger = float("-inf")
        self.last_seen = now


def _scale(person):
    """Shoulder width in normalised units; fall back to a fraction of body-box height."""
    if pv.keypoint_ok(person, 5) and pv.keypoint_ok(person, 6):
        d = math.dist(person.kpts[5], person.kpts[6])
        if d > 0.02:
            return d
    return max(0.05, 0.25 * (person.box[3] - person.box[1]))


class InactivityDetector:
    def __init__(self, seconds=INACTIVITY_SECONDS):
        self.seconds = seconds
        self.states = {}             # track_id -> _State

    def update(self, people, now):
        """Returns (events, idle) where events = [(person, idle_seconds)] and
        idle = {track_id: seconds_since_last_movement}."""
        events, idle = [], {}
        for p in people:
            st = self.states.get(p.track_id)
            if st is None:
                st = self.states[p.track_id] = _State(now)
            st.last_seen = now

            cur = {j: tuple(p.kpts[j]) for j in JOINTS if pv.keypoint_ok(p, j)}
            common = [j for j in cur if j in st.anchor]
            if len(common) >= MIN_COMMON_JOINTS:
                move = sum(math.dist(cur[j], st.anchor[j]) for j in common) / len(common)
                if move / _scale(p) > MOVEMENT_THRESHOLD:
                    st.last_activity, st.fired = now, False
                    st.anchor = cur                      # re-anchor on the new pose
            else:
                for j, xy in cur.items():                # learn joints we have not seen yet
                    st.anchor.setdefault(j, xy)

            seconds_idle = now - st.last_activity
            idle[p.track_id] = seconds_idle
            if (seconds_idle >= self.seconds and not st.fired
                    and now - st.last_trigger >= COOLDOWN_SECONDS):
                st.fired, st.last_trigger = True, now
                events.append((p, seconds_idle))

        for tid in [t for t, s in self.states.items() if now - s.last_seen > FORGET_AFTER_SECONDS]:
            del self.states[tid]
        return events, idle


def run(on_inactive, camera_index=0, show=True):
    tracker, detector = pv.PeopleTracker(), InactivityDetector()
    cap = pv.open_camera(camera_index)
    window_name = "Inactivity (YOLO) - Q to quit"
    if show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, WINDOW_DIMENSIONS)
    while cap is not None and cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        now = time.monotonic()
        people = tracker.process(frame)
        events, idle = detector.update(people, now)
        for p, secs in events:
            print(f">>> person {p.number} INACTIVE for {secs:.0f}s -> trigger")
            on_inactive(p.number, secs)
        if show:
            pv.draw_people(frame, people,
                           notes={p.track_id: [f"idle {idle.get(p.track_id, 0):.0f}s"] for p in people})
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run(lambda person, secs: print(f"(test) person {person} inactive"))
