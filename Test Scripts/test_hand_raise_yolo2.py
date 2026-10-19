"""
Multi-person raised-hand detector using YOLO pose (Ultralytics) with built-in tracking.
More reliable than MediaPipe when 3+ people are in view, small, or partly hidden.

Same interface as hand_raise_multi.py:
    run(on_raise)   ->   on_raise(person_number, side)   side = 'left' | 'right' | 'both'

Install:   python -m pip install ultralytics
First run downloads the yolov8n-pose.pt model (needs internet once).
Use "yolov8s-pose.pt" (slower, more accurate) if people are small or far away.

Note: Ultralytics is licensed AGPL-3.0, fine for coursework; mention it in your report.
"""
import time
from dataclasses import dataclass

import cv2
from ultralytics import YOLO

# ---------------- Configuration ----------------
MODEL_NAME = "yolov8n-pose.pt"   # or yolov8s-pose.pt for better accuracy
MAX_PEOPLE = 5
NUMBERING = "tracked"            # "tracked" or "left_to_right" (leftmost in IMAGE = 1)
DETECT_CONFIDENCE = 0.3          # lower = finds more people
IMAGE_SIZE = 640                 # inference size; try 960 or 1280 for distant people
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
WINDOW_DIMENSIONS = (640, 360)

MARGIN = 0.05                    # wrist above shoulder by this fraction of image height
HOLD_SECONDS = 0.7
COOLDOWN_SECONDS = 5             # per person
MIN_KEYPOINT_CONF = 0.5
MAX_MISSING_FRAMES = 15
# -----------------------------------------------

# COCO keypoint indices used by YOLO pose
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10

SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

PALETTE = [(0, 255, 0), (255, 128, 0), (0, 165, 255), (255, 0, 255), (0, 255, 255)]


def raised_state(kpts, confs):
    """kpts: (17,2) normalised xy; confs: (17,) or None. Returns (label, wrist_ids)."""
    raised = []
    for side, wrist, shoulder in (("left", L_WRIST, L_SHOULDER),
                                  ("right", R_WRIST, R_SHOULDER)):
        if confs is not None and (confs[wrist] < MIN_KEYPOINT_CONF
                                  or confs[shoulder] < MIN_KEYPOINT_CONF):
            continue
        if kpts[wrist][1] == 0 and kpts[wrist][0] == 0:      # undetected keypoint
            continue
        if kpts[shoulder][1] - kpts[wrist][1] > MARGIN:      # wrist above shoulder
            raised.append((side, wrist))
    if not raised:
        return None, []
    sides = [s for s, _ in raised]
    return ("both" if len(sides) == 2 else sides[0]), [w for _, w in raised]


@dataclass
class PersonState:
    number: int
    missing: int = 0
    raised_since: float = None
    raised_label: str = None
    triggered: bool = False
    last_trigger: float = float("-inf")

    def reset_raise(self):
        self.raised_since = None
        self.raised_label = None
        self.triggered = False


class NumberAllocator:
    """Maps YOLO's ever-growing track IDs to small person numbers 1..MAX_PEOPLE."""

    def __init__(self):
        self.by_track = {}                     # track_id -> PersonState

    def assign(self, track_ids):
        for tid in track_ids:
            if tid not in self.by_track:
                used = {p.number for p in self.by_track.values()}
                free = [n for n in range(1, MAX_PEOPLE + 1) if n not in used]
                if free:
                    self.by_track[tid] = PersonState(number=free[0])
        for tid in list(self.by_track):        # age out people not seen this frame
            if tid in track_ids:
                self.by_track[tid].missing = 0
            else:
                st = self.by_track[tid]
                st.missing += 1
                st.reset_raise()
                if st.missing > MAX_MISSING_FRAMES:
                    del self.by_track[tid]


def draw_person(frame, kpts, confs, number, label, wrist_ids):
    h, w = frame.shape[:2]
    colour = PALETTE[(number - 1) % len(PALETTE)]

    def ok(i):
        return (confs is None or confs[i] >= MIN_KEYPOINT_CONF) and (kpts[i][0] or kpts[i][1])

    for a, b in SKELETON:
        if ok(a) and ok(b):
            cv2.line(frame, (int(kpts[a][0] * w), int(kpts[a][1] * h)),
                     (int(kpts[b][0] * w), int(kpts[b][1] * h)), colour, 2)
    for i in range(17):
        if ok(i):
            cv2.circle(frame, (int(kpts[i][0] * w), int(kpts[i][1] * h)), 3, colour, -1)
    for wid in wrist_ids:
        cv2.circle(frame, (int(kpts[wid][0] * w), int(kpts[wid][1] * h)), 14, colour, 3)

    anchor = NOSE if ok(NOSE) else L_SHOULDER
    x = max(5, min(int(kpts[anchor][0] * w) - 40, w - 200))
    y = max(25, int(kpts[anchor][1] * h) - 30)
    text = f"P{number}" + (f": {label.upper()} hand" if label else "")
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)


def run(on_raise, camera_index=0, show=True):
    """Call on_raise(person_number, side) when someone sustains a hand raise."""
    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Create a manually resizable webcam window
    WINDOW_NAME = "YOLO Multi-person Hand Raise - Press Q to Quit"
    if show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_DIMENSIONS)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    print(f"Camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, model: {MODEL_NAME}")

    people = NumberAllocator()
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("Failed to read camera frame.")
                break
            now = time.monotonic()

            result = model.track(frame, persist=True, tracker="bytetrack.yaml",
                                 conf=DETECT_CONFIDENCE, imgsz=IMAGE_SIZE,
                                 max_det=MAX_PEOPLE, verbose=False)[0]

            detections = []          # (track_id, kpts, confs, centre_x)
            n_found = 0
            if result.boxes is not None and result.keypoints is not None:
                n_found = len(result.boxes)
                if result.boxes.id is not None:
                    ids = result.boxes.id.int().cpu().tolist()
                    xyn = result.keypoints.xyn.cpu().numpy()
                    conf = (result.keypoints.conf.cpu().numpy()
                            if result.keypoints.conf is not None else [None] * len(ids))
                    for tid, k, c in zip(ids, xyn, conf):
                        detections.append((tid, k, c, float(k[[L_SHOULDER, R_SHOULDER], 0].mean())))

            people.assign({d[0] for d in detections})

            if NUMBERING == "left_to_right":
                order = sorted(detections, key=lambda d: d[3])
                display = {d[0]: rank + 1 for rank, d in enumerate(order)}
            else:
                display = {tid: people.by_track[tid].number
                           for tid, *_ in detections if tid in people.by_track}

            summary = []
            for tid, kpts, confs, _ in detections:
                if tid not in people.by_track:
                    continue                         # more than MAX_PEOPLE
                state = people.by_track[tid]
                number = display[tid]
                label, wrist_ids = raised_state(kpts, confs)

                if label is None:
                    state.reset_raise()
                else:
                    summary.append(f"P{number} {label}")
                    if state.raised_label != label:
                        state.raised_label, state.raised_since = label, now
                        state.triggered = False
                    if (now - state.raised_since >= HOLD_SECONDS
                            and not state.triggered
                            and now - state.last_trigger >= COOLDOWN_SECONDS):
                        state.last_trigger, state.triggered = now, True
                        print(f"HAND RAISED: person {number}, {label} hand -> trigger")
                        on_raise(number, label)

                if show:
                    draw_person(frame, kpts, confs, number, label, wrist_ids)

            if show:
                status = ("Raised: " + ", ".join(summary)) if summary else "No hands raised"
                line = f"Detected: {n_found}  Tracked: {len(detections)}   {status}"
                cv2.putText(frame, line, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(frame, line, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0) if summary else (255, 255, 255), 2)
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run(lambda person, side: print(f"(test) person {person} raised {side} hand"))
