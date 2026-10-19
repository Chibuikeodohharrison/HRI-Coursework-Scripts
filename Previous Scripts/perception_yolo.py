"""
Shared YOLO-pose perception for the classroom.

ONE YOLO inference per camera frame finds and tracks everyone (up to MAX_PEOPLE).
Each detector (hand raise, inactivity, confusion) then works PER PERSON on that result,
so several detectors can run on one camera without fighting over it.

  PeopleTracker.process(frame)  ->  list[Person]   (stable number per person)
  HandRaiseDetector             ->  per-person raised-hand events
  head_crop(frame, person)      ->  cropped face image (for the expression detector)
  draw_people(...)              ->  skeletons, person numbers and notes on the frame

Install:  python -m pip install ultralytics
Ultralytics is AGPL-3.0 (fine for coursework; mention it in your report).
"""
import time
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO


# ---------------- Configuration ----------------
MODEL_NAME = "yolov8n-pose.pt"   # yolov8s-pose.pt = slower but better for small/far people
MAX_PEOPLE = 8                   # 5 or more: raise as needed (cost grows with people)
NUMBERING = "tracked"            # "tracked" or "left_to_right" (leftmost in the IMAGE = 1)
DETECT_CONFIDENCE = 0.3
IMAGE_SIZE = 640                 # try 960 / 1280 if people are small
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
MAX_MISSING_FRAMES = 15          # forget a person after this many frames unseen
MIN_KEYPOINT_CONF = 0.5
MIN_FACE_PIXELS = 40             # skip faces smaller than this (too small to read)

# Hand raise
RAISE_MARGIN = 0.05              # wrist above shoulder by this fraction of image height
RAISE_HOLD = 0.7                 # seconds the hand must stay up
RAISE_COOLDOWN = 5               # seconds between triggers, per person
# -----------------------------------------------

SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
PALETTE = [(0, 255, 0), (255, 128, 0), (0, 165, 255), (255, 0, 255),
           (0, 255, 255), (255, 255, 0), (128, 128, 255), (200, 200, 200)]


@dataclass
class Person:
    track_id: int          # YOLO's internal ID (unique, never reused): use for per-person STATE
    number: int            # small display number 1..N: use for SPEECH and on-screen labels
    kpts: np.ndarray       # (17, 2) keypoints, normalised 0-1
    confs: np.ndarray      # (17,) keypoint confidences
    box: tuple             # (x1, y1, x2, y2) normalised 0-1


def keypoint_ok(person, i):
    k = person.kpts[i]
    return person.confs[i] >= MIN_KEYPOINT_CONF and (k[0] != 0 or k[1] != 0)


class _Numbering:
    """Maps YOLO's ever-growing track IDs to small numbers, reusing freed numbers."""

    def __init__(self):
        self.numbers, self.missing = {}, {}

    def update(self, ids):
        for tid in ids:
            if tid not in self.numbers:
                used, n = set(self.numbers.values()), 1
                while n in used:
                    n += 1
                self.numbers[tid] = n
            self.missing[tid] = 0
        for tid in list(self.numbers):
            if tid not in ids:
                self.missing[tid] += 1
                if self.missing[tid] > MAX_MISSING_FRAMES:
                    del self.numbers[tid], self.missing[tid]


class PeopleTracker:
    """Runs YOLO pose + ByteTrack once per frame and returns numbered people."""

    def __init__(self, model_name=MODEL_NAME):
        self.model = YOLO(model_name)
        self.numbering = _Numbering()

    def process(self, frame):
        r = self.model.track(frame, persist=True, tracker="bytetrack.yaml",
                             conf=DETECT_CONFIDENCE, imgsz=IMAGE_SIZE,
                             max_det=MAX_PEOPLE, verbose=False)[0]
        people = []
        if r.boxes is not None and r.keypoints is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().cpu().tolist()
            xyn = r.keypoints.xyn.cpu().numpy()
            conf = (r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None
                    else np.ones((len(ids), 17)))
            boxes = r.boxes.xyxyn.cpu().numpy()
            self.numbering.update(set(ids))
            for tid, k, c, b in zip(ids, xyn, conf, boxes):
                people.append(Person(tid, self.numbering.numbers[tid], k, c, tuple(b)))
        else:
            self.numbering.update(set())

        if NUMBERING == "left_to_right":
            for rank, p in enumerate(sorted(people, key=lambda p: (p.box[0] + p.box[2]) / 2)):
                p.number = rank + 1
        return people


def open_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    print(f"Camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    return cap


# ---------------- Hand raise (per person) ----------------
# COCO keypoint indices used by YOLO pose
L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 5, 6, 9, 10


def raised_state(kpts, confs):
    """kpts: (17,2) normalised xy. Returns (label, wrist_ids): label is
    'left', 'right', 'both' or None (the PERSON'S own hand)."""
    raised = []
    for side, wrist, shoulder in (("left", L_WRIST, L_SHOULDER),
                                  ("right", R_WRIST, R_SHOULDER)):
        if confs is not None and (confs[wrist] < MIN_KEYPOINT_CONF
                                  or confs[shoulder] < MIN_KEYPOINT_CONF):
            continue
        if kpts[wrist][0] == 0 and kpts[wrist][1] == 0:          # undetected keypoint
            continue
        if kpts[shoulder][1] - kpts[wrist][1] > RAISE_MARGIN:    # wrist above shoulder
            raised.append((side, wrist))
    if not raised:
        return None, []
    sides = [s for s, _ in raised]
    return ("both" if len(sides) == 2 else sides[0]), [w for _, w in raised]


class HandRaiseDetector:
    """Per-person sustained hand raise -> events (person, side)."""

    def __init__(self):
        self.state = {}      # track_id -> dict

    def update(self, people, now):
        events, highlight = [], {}
        for p in people:
            st = self.state.setdefault(p.track_id, {"label": None, "since": None,
                                                    "triggered": False, "last": float("-inf")})
            label, wrists = raised_state(p.kpts, p.confs)
            if label is None:
                st.update(label=None, since=None, triggered=False)
                continue
            highlight[p.track_id] = (label, wrists)
            if st["label"] != label:
                st.update(label=label, since=now, triggered=False)
            if (now - st["since"] >= RAISE_HOLD and not st["triggered"]
                    and now - st["last"] >= RAISE_COOLDOWN):
                st.update(triggered=True, last=now)
                events.append((p, label))
        return events, highlight


# ---------------- Face crop for the expression detector ----------------
def head_crop(frame, person):
    """Crop the head region from the YOLO face keypoints (0 nose, 1-2 eyes, 3-4 ears).
    Returns a BGR image, or None if the face is not visible or too small."""
    h, w = frame.shape[:2]
    idx = [i for i in range(5) if keypoint_ok(person, i)]
    if len(idx) < 3:
        return None
    pts = np.array([(person.kpts[i][0] * w, person.kpts[i][1] * h) for i in idx])
    centre = pts.mean(axis=0)
    span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
    side = span * 1.9
    if keypoint_ok(person, 1) and keypoint_ok(person, 2):      # eye distance fallback
        eye = np.hypot((person.kpts[1][0] - person.kpts[2][0]) * w,
                       (person.kpts[1][1] - person.kpts[2][1]) * h)
        side = max(side, eye * 4.5)
    if side < MIN_FACE_PIXELS:
        return None
    x1, y1 = int(max(0, centre[0] - side / 2)), int(max(0, centre[1] - side / 2))
    x2, y2 = int(min(w, centre[0] + side / 2)), int(min(h, centre[1] + side / 2))
    if x2 - x1 < MIN_FACE_PIXELS or y2 - y1 < MIN_FACE_PIXELS:
        return None
    crop = frame[y1:y2, x1:x2]
    longest = max(crop.shape[:2])
    if longest < 256:                                          # upscale small faces
        crop = cv2.resize(crop, None, fx=256 / longest, fy=256 / longest,
                          interpolation=cv2.INTER_CUBIC)
    return crop


# ---------------- Drawing ----------------
def draw_people(frame, people, notes=None, highlight=None):
    """notes: {track_id: [text lines]}  highlight: {track_id: (label, wrist_ids)}"""
    notes, highlight = notes or {}, highlight or {}
    h, w = frame.shape[:2]
    for p in people:
        colour = PALETTE[(p.number - 1) % len(PALETTE)]
        for a, b in SKELETON:
            if keypoint_ok(p, a) and keypoint_ok(p, b):
                cv2.line(frame, (int(p.kpts[a][0] * w), int(p.kpts[a][1] * h)),
                         (int(p.kpts[b][0] * w), int(p.kpts[b][1] * h)), colour, 2)
        for i in range(17):
            if keypoint_ok(p, i):
                cv2.circle(frame, (int(p.kpts[i][0] * w), int(p.kpts[i][1] * h)), 3, colour, -1)
        if p.track_id in highlight:
            for wid in highlight[p.track_id][1]:
                cv2.circle(frame, (int(p.kpts[wid][0] * w), int(p.kpts[wid][1] * h)), 14, colour, 3)

        anchor = 0 if keypoint_ok(p, 0) else (5 if keypoint_ok(p, 5) else None)
        if anchor is None:
            x, y = int(p.box[0] * w), int(p.box[1] * h) + 20
        else:
            x, y = int(p.kpts[anchor][0] * w) - 40, int(p.kpts[anchor][1] * h) - 30
        x, y = max(5, min(x, w - 230)), max(25, y)
        lines = [f"P{p.number}"] + list(notes.get(p.track_id, []))
        for n, text in enumerate(lines):
            yy = y + n * 22
            cv2.putText(frame, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(frame, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)


if __name__ == "__main__":
    tracker, hands = PeopleTracker(), HandRaiseDetector()
    cap = open_camera()
    while cap is not None and cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        people = tracker.process(frame)
        events, hl = hands.update(people, time.monotonic())
        for p, side in events:
            print(f"HAND RAISED: person {p.number}, {side}")
        draw_people(frame, people, highlight=hl)
        cv2.imshow("People tracking (Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
