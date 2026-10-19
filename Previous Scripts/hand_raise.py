"""
Raised-hand detector using the newer MediaPipe Tasks API.

A hand is raised when its wrist is above its shoulder
by a specified margin for HOLD_SECONDS.

Triggers once per gesture, with a cooldown.

CHANGE FROM PREVIOUS VERSION:
  hand_is_raised() now returns "left", "right" or None (instead of True/False)
  run() now calls on_raise(side) with the side that was raised.
  Sides are the PERSON'S own left/right (MediaPipe anatomical labels).
"""

import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp


# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


# Configuration
MARGIN = 0.05
HOLD_SECONDS = 0.7
COOLDOWN_SECONDS = 5
MIN_VISIBILITY = 0.5

MODEL_PATH = Path(__file__).resolve().parent / "pose_landmarker_lite.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/"
    "pose_landmarker_lite.task"
)

# MediaPipe Pose landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16

# Skeleton connections
POSE_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
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


def hand_is_raised(landmarks):
    """Return 'left', 'right' or None.

    If both hands are up, the higher one (smaller y) wins.
    """

    pairs = [
        ("left", LEFT_WRIST, LEFT_SHOULDER),
        ("right", RIGHT_WRIST, RIGHT_SHOULDER),
    ]

    best_side = None
    best_height = 0.0  # how far the wrist is above the shoulder

    for side, wrist_id, shoulder_id in pairs:
        wrist = landmarks[wrist_id]
        shoulder = landmarks[shoulder_id]

        if (
            wrist.visibility < MIN_VISIBILITY
            or shoulder.visibility < MIN_VISIBILITY
        ):
            continue

        height = shoulder.y - wrist.y  # positive = wrist above shoulder
        if height > MARGIN and height > best_height:
            best_side = side
            best_height = height

    return best_side


def draw_pose(frame, landmarks):
    """Draw pose landmarks and skeleton using OpenCV."""

    height, width = frame.shape[:2]

    for start_id, end_id in POSE_CONNECTIONS:
        start = landmarks[start_id]
        end = landmarks[end_id]

        if (
            start.visibility < MIN_VISIBILITY
            or end.visibility < MIN_VISIBILITY
        ):
            continue

        p1 = (int(start.x * width), int(start.y * height))
        p2 = (int(end.x * width), int(end.y * height))

        cv2.line(frame, p1, p2, (0, 255, 0), 2)

    for landmark in landmarks:
        if landmark.visibility < MIN_VISIBILITY:
            continue

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        if 0 <= x < width and 0 <= y < height:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


def run(on_raise, camera_index=0, show=True):
    """Call on_raise(side) when a sustained hand raise is detected.

    side is 'left' or 'right' (the person's own hand).
    on_raise should return quickly (start a thread for slow work such as
    moving the robot) so the camera loop does not freeze.
    """

    ensure_model()

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    raised_since = None
    last_trigger = float("-inf")
    triggered_for_current_raise = False

    start_time = time.monotonic()
    last_timestamp_ms = -1

    try:
        with PoseLandmarker.create_from_options(options) as landmarker:

            while cap.isOpened():
                ok, frame = cap.read()

                if not ok:
                    print("Failed to read camera frame.")
                    break

                # Convert BGR to RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Create MediaPipe image
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                # Generate monotonically increasing timestamp
                now = time.monotonic()
                timestamp_ms = max(
                    int((now - start_time) * 1000),
                    last_timestamp_ms + 1
                )
                last_timestamp_ms = timestamp_ms

                # Run pose detection
                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                status = "No person detected"

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]

                    side = hand_is_raised(landmarks)

                    if side is not None:
                        status = f"Hand raised ({side})"

                        if raised_since is None:
                            raised_since = now

                        held = now - raised_since

                        if (
                            held >= HOLD_SECONDS
                            and not triggered_for_current_raise
                            and now - last_trigger >= COOLDOWN_SECONDS
                        ):
                            last_trigger = now
                            triggered_for_current_raise = True

                            print(f"HAND RAISED ({side}) -> trigger")
                            on_raise(side)

                    else:
                        status = "Hand down"
                        raised_since = None
                        triggered_for_current_raise = False

                    if show:
                        draw_pose(frame, landmarks)

                else:
                    raised_since = None
                    triggered_for_current_raise = False

                if show:
                    color = (
                        (0, 255, 0)
                        if status.startswith("Hand raised")
                        else (0, 0, 255)
                    )

                    cv2.putText(
                        frame,
                        status,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2
                    )

                    cv2.imshow(
                        "Hand Raise Detector - Press Q to Quit",
                        frame
                    )

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run(lambda side: print("(test) trigger fired:", side))
