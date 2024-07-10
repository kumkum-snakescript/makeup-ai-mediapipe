import cv2 

import cv2
import math
import numpy as np

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  cv2.imshow('img', img)

import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

vid = cv2.VideoCapture(0) 

while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
    image = cv2.flip(frame, 1)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=2,
        min_detection_confidence=0.5) as face_mesh:
    # for name, image in images.items():
    # Convert the BGR image to RGB and process it with MediaPipe Face Mesh.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        print(results)

    # Draw face landmarks of each face.
        # print(f'Face landmarks of {name}:')
        if not results.multi_face_landmarks:
            continue
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())
            resize_and_show(annotated_image)

            cv2.imshow('frame', annotated_image) 

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 