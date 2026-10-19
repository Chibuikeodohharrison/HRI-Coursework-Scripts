"""
Expression category + score detector using MediaPipe Face Landmarker blendshapes.

Each category is a weighted sum of blendshapes, measured as the excess over the
student's own neutral baseline. The category with the highest score relative to
its threshold is reported. Only "confused-like" can trigger the robot, and it
must be sustained for HOLD_SECONDS. All weights/thresholds are heuristic guesses.
"""
import os
import time
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = "face_landmarker.task"

# category -> ({blendshape: weight}, threshold for that category's score)
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
MILD_THRESHOLD = 0.05        # confused-like score between this and its threshold = "mild tension"

ALL_SHAPES = sorted({n for w, _ in CATEGORIES.values() for n in w})

CALIBRATION_SECONDS = 5
HOLD_SECONDS = 2.0           # confused-like must persist this long to trigger
COOLDOWN_SECONDS = 20
SMOOTHING = 0.3              # EMA factor (higher = more responsive, noisier)
PRINT_EVERY = 0.5            # seconds between printed lines


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading face model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def get_scores(result):
    if not result.face_blendshapes:
        return None
    return {c.category_name: c.score for c in result.face_blendshapes[0]}


def category_score(scores, baseline, weights):
    """Weighted excess over baseline, normalised by total weight (roughly 0-1)."""
    total = sum(w * max(0.0, scores.get(n, 0.0) - baseline.get(n, 0.0))
                for n, w in weights.items())
    return total / sum(weights.values())


def classify(smoothed):
    """Return (label, score_of_label) from smoothed per-category scores."""
    # Pick the category that most exceeds its own threshold
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


def run(on_confused, camera_index=0, show=True, log_scores=None):
    """
    on_confused(): called once when a sustained confused-like expression is seen.
    log_scores(timestamp, label, score, all_scores): optional per-frame logger
    for choosing thresholds from your own test recordings.
    """
    ensure_model()
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=True,
        num_faces=1)
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(camera_index)
    start = time.time()
    calib_samples, baseline = [], None
    smoothed = {name: 0.0 for name in CATEGORIES}
    above_since, last_trigger, last_print = None, 0, 0
    label, score = "no face", 0.0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, int((now - start) * 1000))
        scores = get_scores(result)

        if scores is None:
            label, score, above_since = "no face", 0.0, None

        elif baseline is None:                      # Phase 1: calibration
            calib_samples.append(scores)
            label, score = "calibrating (neutral face)", 0.0
            if now - start >= CALIBRATION_SECONDS:
                baseline = {n: sum(s.get(n, 0) for s in calib_samples) / len(calib_samples)
                            for n in ALL_SHAPES}
                print("Baseline recorded.")

        else:                                       # Phase 2: detection
            for name, (weights, _) in CATEGORIES.items():
                raw = category_score(scores, baseline, weights)
                smoothed[name] = SMOOTHING * raw + (1 - SMOOTHING) * smoothed[name]
            label, score = classify(smoothed)

            if log_scores:
                log_scores(now, label, score, dict(smoothed))

            if label == CONFUSED:
                if above_since is None:
                    above_since = now
                if (now - above_since >= HOLD_SECONDS
                        and now - last_trigger > COOLDOWN_SECONDS):
                    last_trigger, above_since = now, None
                    print(">>> CONFUSION-LIKE EXPRESSION SUSTAINED -> trigger")
                    on_confused()
            else:
                above_since = None

        # Print category + score at a readable rate
        if now - last_print >= PRINT_EVERY:
            last_print = now
            if baseline is not None and scores is not None:
                detail = "  ".join(f"{n}={v:.2f}" for n, v in smoothed.items())
                print(f"{label:<36} score={score:.2f} | {detail}")
            else:
                print(label)

        if show:
            cv2.putText(frame, f"{label}  {score:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Expression detector (q quit, r recalibrate)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                baseline, calib_samples, start = None, [], time.time()
                smoothed = {name: 0.0 for name in CATEGORIES}

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    run(lambda: print("(test) confused cue fired"))