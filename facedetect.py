import cv2
import numpy as np

# Load the Haar cascade
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = detect_faces(frame)
        cv2.imshow('Facial Detection', frame)

        # Increase wait time to ensure it registers 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("Exiting...")
            break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
