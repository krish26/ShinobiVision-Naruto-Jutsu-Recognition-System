import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.8,
    min_hand_presence_confidence=0.8,
    min_tracking_confidence=0.8
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = None, None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    results = detector.detect(mp_image)

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:

            # Get important landmarks
            index_tip = hand_landmarks[8]
            index_pip = hand_landmarks[6]

            middle_tip = hand_landmarks[12]
            middle_pip = hand_landmarks[10]

            ring_tip = hand_landmarks[16]
            ring_pip = hand_landmarks[14]

            pinky_tip = hand_landmarks[20]
            pinky_pip = hand_landmarks[18]

            # Check if fingers are up
            index_up = index_tip.y < index_pip.y
            middle_up = middle_tip.y < middle_pip.y
            ring_up = ring_tip.y < ring_pip.y
            pinky_up = pinky_tip.y < pinky_pip.y

            x, y = int(index_tip.x * w), int(index_tip.y * h)

            # ✋ Draw only if ONLY index finger is up
            if index_up and not middle_up and not ring_up and not pinky_up:

                if prev_x is None:
                    prev_x, prev_y = x, y

                cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 255), 5)
                prev_x, prev_y = x, y

            else:
                # Stop drawing if fist or other gesture
                prev_x, prev_y = None, None

    else:
        prev_x, prev_y = None, None

    frame = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)

    cv2.imshow("Gesture Drawing", frame)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)

cap.release()
cv2.destroyAllWindows()