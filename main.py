import cv2
import math
import numpy as np
import mediapipe as mp

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

def resize_and_show(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
    cv2.imshow('img', img)

def apply_blush(image, landmarks, indices, color=(0, 0, 255), alpha=0.3):
    overlay = image.copy()
    points = []
    for idx in indices:
        landmark = landmarks[idx]
        x = int(landmark.x * image.shape[1])
        y = int(landmark.y * image.shape[0])
        points.append((x, y))
    if points:
        points = np.array(points, dtype=np.int32)
        cv2.fillConvexPoly(overlay, points, color)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=2,
    min_detection_confidence=0.5)

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

vid = cv2.VideoCapture(0)

while vid.isOpened():
    _, frame = vid.read() 
    image = cv2.flip(frame, 1)

    # Face Mesh Detection
    face_results = mp_face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            left_cheek_indices = [143, 227, 123, 187, 207, 101, 118, 117, 111]
            right_cheek_indices = [372, 340, 346, 330, 427, 352, 345]
            outer_lips_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
            inner_lips_indices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
            
            # Apply blush to the lips
            apply_blush(image, face_landmarks.landmark, outer_lips_indices, color=(0, 0, 255), alpha=0.3)
            apply_blush(image, face_landmarks.landmark, inner_lips_indices, color=(0, 0, 255), alpha=0.3)

            # Apply blush to the cheeks
            apply_blush(image, face_landmarks.landmark, left_cheek_indices, color=(0, 0, 255), alpha=0.1)
            apply_blush(image, face_landmarks.landmark, right_cheek_indices, color=(0, 0, 255), alpha=0.1)

            # Hand Detection
            hand_results = mp_hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

        cv2.imshow('frame', image)
    
    if cv2.waitKey(1) & 0xFF == ord('x'): 
        break

vid.release() 
cv2.destroyAllWindows()
