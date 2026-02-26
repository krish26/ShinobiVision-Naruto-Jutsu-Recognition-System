import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import random
from collections import deque

# HandLandmarker setup
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

# Game setup
screen_w, screen_h = 640, 480
snake = deque()
snake.append((screen_w//2, screen_h//2))
snake_length = 5
food = (random.randint(50, screen_w-50), random.randint(50, screen_h-50))
score = 0
snake_color = (0, 255, 0)
food_color = (0, 0, 255)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    results = detector.detect(mp_image)

    finger_x, finger_y = snake[-1]

    if results.hand_landmarks:
        hand_landmarks = results.hand_landmarks[0]
        index_tip = hand_landmarks[8]
        finger_x = int(index_tip.x * w * screen_w / w)
        finger_y = int(index_tip.y * h * screen_h / h)

    snake.append((finger_x, finger_y))
    if len(snake) > snake_length:
        snake.popleft()

    if abs(finger_x - food[0]) < 20 and abs(finger_y - food[1]) < 20:
        score += 1
        snake_length += 5
        food = (random.randint(50, screen_w-50), random.randint(50, screen_h-50))

    game_frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    cv2.circle(game_frame, food, 10, food_color, -1)
    for x, y in snake:
        cv2.circle(game_frame, (x, y), 10, snake_color, -1)

    cv2.putText(game_frame, f"Score: {score}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Finger Snake Game", game_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()