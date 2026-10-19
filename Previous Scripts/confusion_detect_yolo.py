"""
Per-person CONFUSION-LIKE expression detector for many students.

YOLO pose (perception_yolo.py) finds and tracks everyone; for each person we crop the head
and run MediaPipe Face Landmarker on that crop. The scoring, categories, thresholds and
hold/cooldown logic come from your original confusion_detect2.py design, but every person gets their
OWN neutral baseline, smoothing and timers.

  on_confused(person_number)   fires once per sustained confused-like expression.

Calibration: each new person is calibrated from their first CALIBRATION_SECONDS of
readable face frames, so ask students to look neutral when they first appear.
Press R in the window to recalibrate everyone.

This is a heuristic "confusion-like expression cue", not validated emotion recognition.
Frames are processed in memory only; nothing is stored.

Needs: perception_yolo.py in the same folder (no other detector files).
"""
import time

import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import perception_yolo as pv

# ---------------- Configuration ----------------
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = "face_landmarker.task"

# category -> ({blendshape: weight}, threshold for that category's score)
# Heuristic guesses: tune them on your own recordings.
CATEGORIES = {
    "confused-like": ({"browDownLeft": 1.0, "browDownRight": 1.0,
                       "browInnerUp": 0.8,
                       "eyeSquintLeft": 0.5, "eyeSquintRight": 0.5,
                       "mouthPressLeft": 0.4, "mouthPressRight": 0.4}, 0.12),
    "smiling":       ({"mouthSmileLeft": 1.0, "mouthSmileRight": 1.0}, 0.25),
    "surprised":     ({"eyeWideLeft": 1.0, "eyeWideRight": 1.0,
                       "browOuterUpLeft": 0.7, "browOuterUpRight": 0.7,
                       "jawOpen": 0.8}, 0.20),
    "frowning/sad":  ({"mouthFrownLeft": 1.0, "mouthFrownRight": 1.0}, 0.20),
}
CONFUSED = "confused-like"
MILD_THRESHOLD = 0.05            # confused-like score between this and its threshold = mild tension
ALL_SHAPES = sorted({n for w, _ in CATEGORIES.values() for n in w})

CALIBRATION_SECONDS = 5
HOLD_SECONDS = 2.0               # confused-like must persist this long to trigger
COOLDOWN_SECONDS = 20            # per person
SMOOTHING = 0.3                  # EMA factor (higher = more responsive, noisier)
MIN_SECONDS_BETWEEN_FACE_CHECKS = 0.15   # per person; limits CPU use with many faces
FORGET_AFTER_SECONDS = 10
# -----------------------------------------------


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading face model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def get_scores(result):
    if not result.face_blendshapes:
        return None
    return {c.category_name: c.score for c in result.face_blendshapes[0]}


def category_score(scores, baseline, weights):
    """Weighted excess over the person's baseline, normalised by total weight (roughly 0-1)."""
    total = sum(w * max(0.0, scores.get(n, 0.0) - baseline.get(n, 0.0))
                for n, w in weights.items())
    return total / sum(weights.values())


def classify(smoothed):
    """Return (label, score_of_label) from smoothed per-category scores."""
    best, best_ratio = None, 1.0
    for name, (_, thr) in CATEGORIES.items():
        ratio = smoothed[name] / thr
        if ratio >= best_ratio:
            best, best_ratio = name, ratio
    if best is not None:
        return best, smoothed[best]
    if smoothed[CONFUSED] >= MILD_THRESHOLD:
        return "mild tension (maybe concentrating)", smoothed[CONFUSED]
    return "neutral", smoothed[CONFUSED]


class _PersonFace:
    def __init__(self, now):
        self.calib_start = now
        self.calib_samples = []
        self.baseline = None
        self.smoothed = {n: 0.0 for n in CATEGORIES}
        self.label, self.score = "calibrating", 0.0
        self.above_since = None
        self.last_trigger = float("-inf")
        self.last_check = float("-inf")
        self.last_seen = now


class ConfusionDetector:
    def __init__(self):
        ensure_model()
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,       # we feed cropped faces one at a time
            output_face_blendshapes=True,
            num_faces=1)
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.faces = {}                                  # track_id -> _PersonFace

    def recalibrate(self):
        self.faces.clear()

    def update(self, frame, people, now):
        """Returns (events, status): events = [person], status = {track_id: (label, score)}."""
        events, status = [], {}
        for p in people:
            pf = self.faces.setdefault(p.track_id, _PersonFace(now))
            pf.last_seen = now
            status[p.track_id] = (pf.label, pf.score)

            if now - pf.last_check < MIN_SECONDS_BETWEEN_FACE_CHECKS:
                continue
            pf.last_check = now

            crop = pv.head_crop(frame, p)
            if crop is None:
                pf.above_since = None
                pf.label, pf.score = "no face", 0.0
                status[p.track_id] = (pf.label, pf.score)
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            result = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            scores = get_scores(result)
            if scores is None:
                pf.above_since = None
                pf.label, pf.score = "no face", 0.0
                status[p.track_id] = (pf.label, pf.score)
                continue

            if pf.baseline is None:                       # calibration phase
                pf.calib_samples.append(scores)
                pf.label, pf.score = "calibrating", 0.0
                if now - pf.calib_start >= CALIBRATION_SECONDS and len(pf.calib_samples) >= 5:
                    n = len(pf.calib_samples)
                    pf.baseline = {k: sum(s.get(k, 0) for s in pf.calib_samples) / n
                                   for k in ALL_SHAPES}
                    print(f"[confusion] baseline recorded for person {p.number}")
            else:                                         # detection phase
                for name, (weights, _) in CATEGORIES.items():
                    raw = category_score(scores, pf.baseline, weights)
                    pf.smoothed[name] = SMOOTHING * raw + (1 - SMOOTHING) * pf.smoothed[name]
                pf.label, pf.score = classify(pf.smoothed)

                if pf.label == CONFUSED:
                    if pf.above_since is None:
                        pf.above_since = now
                    if (now - pf.above_since >= HOLD_SECONDS
                            and now - pf.last_trigger >= COOLDOWN_SECONDS):
                        pf.last_trigger, pf.above_since = now, None
                        print(f">>> person {p.number}: CONFUSION-LIKE EXPRESSION SUSTAINED -> trigger")
                        events.append(p)
                else:
                    pf.above_since = None
            status[p.track_id] = (pf.label, pf.score)

        for tid in [t for t, f in self.faces.items() if now - f.last_seen > FORGET_AFTER_SECONDS]:
            del self.faces[tid]
        return events, status


def run(on_confused, camera_index=0, show=True):
    tracker, detector = pv.PeopleTracker(), ConfusionDetector()
    cap = pv.open_camera(camera_index)
    while cap is not None and cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        now = time.monotonic()
        people = tracker.process(frame)
        events, status = detector.update(frame, people, now)
        for p in events:
            on_confused(p.number)
        if show:
            pv.draw_people(frame, people, notes={
                p.track_id: [f"{status[p.track_id][0]} {status[p.track_id][1]:.2f}"]
                for p in people if p.track_id in status})
            cv2.imshow("Confusion cue (YOLO + face) - Q quit, R recalibrate", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                detector.recalibrate()
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run(lambda person: print(f"(test) person {person} looks confused"))
