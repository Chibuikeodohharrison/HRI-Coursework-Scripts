"""
Multi-person raised-hand detector (up to MAX_PEOPLE people at once)
using the MediaPipe Tasks API.

For each person it reports WHICH hand is raised ("left", "right" or "both")
and a PERSON NUMBER, both on screen and in the callback:

    on_raise(person_number, side)

A hand counts as raised when its wrist is above its shoulder by MARGIN for
HOLD_SECONDS. Each person has their own timer and cooldown, so five students
can raise hands independently.

Person numbers:
  NUMBERING = "tracked"       -> each person keeps the same number while they stay
                                 in view (numbers are assigned in order of appearance)
  NUMBERING = "left_to_right" -> person 1 is the leftmost in the CAMERA IMAGE,
                                 then 2, 3 ... (can change if people move or enter)

"left"/"right" are the PERSON'S OWN left/right hand (anatomical), so a student
facing the camera raising their left hand appears on the RIGHT of the image.
"""

import math
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp


# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


# ---------------- Configuration ----------------
MAX_PEOPLE = 5
NUMBERING = "tracked"            # "tracked" or "left_to_right"

MARGIN = 0.05                    # wrist must be this far above shoulder (normalised)
HOLD_SECONDS = 0.7               # hand must stay up this long
COOLDOWN_SECONDS = 5             # per person, between triggers
MIN_VISIBILITY = 0.5

MATCH_DISTANCE = 0.15            # max movement between frames to count as same person
MAX_MISSING_FRAMES = 15          # forget a person after this many frames unseen

MODEL_PATH = Path(__file__).resolve().parent / "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/"
    "pose_landmarker_lite.task"
)
# -----------------------------------------------

# MediaPipe Pose landmark indices
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]

# One colour (BGR) per person number 1..5
PALETTE = [
    (0, 255, 0),      # 1 green
    (255, 128, 0),    # 2 blue
    (0, 165, 255),    # 3 orange
    (255, 0, 255),    # 4 magenta
    (0, 255, 255),    # 5 yellow
]


def ensure_model():
    """Download the pose model if it is missing."""
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return
    print("Downloading MediaPipe Pose Landmarker model...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully.")
    except Exception:
        MODEL_PATH.unlink(missing_ok=True)
        raise


def raised_state(landmarks):
    """Return (label, wrist_ids): label is 'left', 'right', 'both' or None."""
    raised = []
    for side, wrist_id, shoulder_id in (
        ("left", LEFT_WRIST, LEFT_SHOULDER),
        ("right", RIGHT_WRIST, RIGHT_SHOULDER),
    ):
        wrist, shoulder = landmarks[wrist_id], landmarks[shoulder_id]
        if wrist.visibility < MIN_VISIBILITY or shoulder.visibility < MIN_VISIBILITY:
            continue
        if shoulder.y - wrist.y > MARGIN:      # positive = wrist above shoulder
            raised.append((side, wrist_id))

    if not raised:
        return None, []
    sides = [s for s, _ in raised]
    label = "both" if len(sides) == 2 else sides[0]
    return label, [w for _, w in raised]


def person_centre(landmarks):
    """Rough body position (x, y) in normalised image coordinates, or None."""
    ls, rs = landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER]
    if ls.visibility >= MIN_VISIBILITY and rs.visibility >= MIN_VISIBILITY:
        return ((ls.x + rs.x) / 2, (ls.y + rs.y) / 2)
    nose = landmarks[NOSE]
    if nose.visibility >= MIN_VISIBILITY:
        return (nose.x, nose.y)
    visible = [p for p in landmarks if p.visibility >= MIN_VISIBILITY]
    if not visible:
        return None
    return (sum(p.x for p in visible) / len(visible),
            sum(p.y for p in visible) / len(visible))


# ---------------- Simple multi-person tracker ----------------
@dataclass
class Track:
    id: int
    centre: tuple
    missing: int = 0
    raised_since: float = None
    raised_label: str = None
    triggered: bool = False
    last_trigger: float = float("-inf")

    def reset_raise(self):
        self.raised_since = None
        self.raised_label = None
        self.triggered = False


class PersonTracker:
    """Keeps the same ID for a person across frames by matching nearest positions."""

    def __init__(self):
        self.tracks = {}                       # id -> Track

    def _free_id(self):
        for i in range(1, MAX_PEOPLE + 1):
            if i not in self.tracks:
                return i
        return None

    def update(self, detections):
        """detections: list of (centre, landmarks). Returns list of (Track, landmarks)."""
        pairs = []
        for tid, track in self.tracks.items():
            for di, (centre, _) in enumerate(detections):
                dist = math.dist(track.centre, centre)
                if dist <= MATCH_DISTANCE:
                    pairs.append((dist, tid, di))
        pairs.sort()

        used_tracks, used_dets, matched = set(), set(), {}
        for _, tid, di in pairs:
            if tid in used_tracks or di in used_dets:
                continue
            used_tracks.add(tid)
            used_dets.add(di)
            matched[di] = tid

        for di, tid in matched.items():
            self.tracks[tid].centre = detections[di][0]
            self.tracks[tid].missing = 0

        # Age (and eventually drop) people who were not matched this frame
        for tid in list(self.tracks):
            if tid not in used_tracks:
                track = self.tracks[tid]
                track.missing += 1
                track.reset_raise()
                if track.missing > MAX_MISSING_FRAMES:
                    del self.tracks[tid]

        # New people
        for di, (centre, _) in enumerate(detections):
            if di in used_dets:
                continue
            new_id = self._free_id()
            if new_id is None:
                continue
            self.tracks[new_id] = Track(new_id, centre)
            matched[di] = new_id

        return [(self.tracks[tid], detections[di][1]) for di, tid in matched.items()]


# ---------------- Drawing ----------------
def draw_person(frame, landmarks, number, label, wrist_ids):
    height, width = frame.shape[:2]
    colour = PALETTE[(number - 1) % len(PALETTE)]

    for a, b in POSE_CONNECTIONS:
        pa, pb = landmarks[a], landmarks[b]
        if pa.visibility < MIN_VISIBILITY or pb.visibility < MIN_VISIBILITY:
            continue
        cv2.line(frame, (int(pa.x * width), int(pa.y * height)),
                 (int(pb.x * width), int(pb.y * height)), colour, 2)

    for p in landmarks:
        if p.visibility < MIN_VISIBILITY:
            continue
        x, y = int(p.x * width), int(p.y * height)
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(frame, (x, y), 3, colour, -1)

    # Highlight the raised wrist(s)
    for wid in wrist_ids:
        w = landmarks[wid]
        cv2.circle(frame, (int(w.x * width), int(w.y * height)), 14, colour, 3)

    # Label above the head/shoulders
    anchor = landmarks[NOSE] if landmarks[NOSE].visibility >= MIN_VISIBILITY \
        else landmarks[LEFT_SHOULDER]
    x = max(5, min(int(anchor.x * width) - 40, width - 200))
    y = max(25, int(anchor.y * height) - 30)
    text = f"P{number}" + (f": {label.upper()} hand" if label else "")
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)


# ---------------- Main loop ----------------
def run(on_raise, camera_index=0, show=True):
    """Call on_raise(person_number, side) when someone sustains a hand raise.

    side is 'left', 'right' or 'both' (the person's own hand).
    on_raise should return quickly (start a thread for slow work such as
    moving the robot) so the camera loop does not freeze.
    """
    ensure_model()

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_poses=MAX_PEOPLE,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    tracker = PersonTracker()
    start_time = time.monotonic()
    last_timestamp_ms = -1

    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    print("Failed to read camera frame.")
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                now = time.monotonic()
                timestamp_ms = max(int((now - start_time) * 1000), last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                # Build (centre, landmarks) for every detected person
                detections = []
                for landmarks in result.pose_landmarks:
                    centre = person_centre(landmarks)
                    if centre is not None:
                        detections.append((centre, landmarks))

                tracked = tracker.update(detections)

                # Decide each person's display number
                if NUMBERING == "left_to_right":
                    order = sorted(tracked, key=lambda tl: tl[0].centre[0])
                    numbers = {t.id: rank + 1 for rank, (t, _) in enumerate(order)}
                else:
                    numbers = {t.id: t.id for t, _ in tracked}

                summary = []
                for track, landmarks in tracked:
                    number = numbers[track.id]
                    label, wrist_ids = raised_state(landmarks)

                    if label is None:
                        track.reset_raise()
                    else:
                        summary.append(f"P{number} {label}")
                        if track.raised_label != label:       # new or changed raise
                            track.raised_label = label
                            track.raised_since = now
                            track.triggered = False

                        held = now - track.raised_since
                        if (held >= HOLD_SECONDS
                                and not track.triggered
                                and now - track.last_trigger >= COOLDOWN_SECONDS):
                            track.last_trigger = now
                            track.triggered = True
                            print(f"HAND RAISED: person {number}, {label} hand -> trigger")
                            on_raise(number, label)

                    if show:
                        draw_person(frame, landmarks, number, label, wrist_ids)

                if show:
                    status = ("Raised: " + ", ".join(summary)) if summary else "No hands raised"
                    cv2.putText(frame, f"People: {len(tracked)}   {status}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                    cv2.putText(frame, f"People: {len(tracked)}   {status}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0) if summary else (255, 255, 255), 2)
                    cv2.imshow("Multi-person Hand Raise Detector - Press Q to Quit", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run(lambda person, side: print(f"(test) person {person} raised {side} hand"))
