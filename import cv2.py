import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np

# Load the pre-trained emotion detection model
emotion_model = load_model('model.h5')

# Load the Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier('face_haarcascades.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Load the background subtraction model
bg_subtraction = cv2.createBackgroundSubtractorMOG2()

# Initialize variables
video = cv2.VideoCapture("video.mp4")
prev_x = 0
prev_y = 0
head_movement_count = 0
eye_blink_count = 0
hand_gesture_count = 0
eye_detected = False

frame_count = 0
frame_interval = 3

while video.isOpened():
    ret, frame = video.read()

    if not ret:
        break

    frame_count += 1

    if frame_count % frame_interval != 0:
        continue

    # Apply background subtraction to detect hand gestures
    fg_mask = bg_subtraction.apply(frame)
    fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]

    # Perform face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=15, minSize=(30, 30))

    if len(face_rect) > 0:
        (x, y, w, h) = face_rect[0]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 5)

        curr_x = x + w // 2
        curr_y = y + h // 2

        if prev_x != 0 and prev_y != 0:
            distance = ((curr_x - prev_x) * 2 + (curr_y - prev_y) * 2) ** 0.5
            if distance > 10:
                head_movement_count += 1

        prev_x = curr_x
        prev_y = curr_y

        # Perform eye detection
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10))

        if len(eyes) > 0:
            eye_detected = True
        else:
            if eye_detected:
                eye_blink_count += 1
                eye_detected = False

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 3)

    # Perform hand gesture recognition
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 5000:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            hand_gesture_count += 1

    cv2.putText(frame, f"Head Movement Count: {head_movement_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"Eye Blink Count: {eye_blink_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Hand Gesture Count: {hand_gesture_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()