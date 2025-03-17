import cv2
import joblib
import numpy as np
import pygame
import threading  # Import the threading module

# Load the trained model
svm_model = joblib.load('drowsiness_detection_svm_model.pkl')

# Open the webcam (or any other camera)
cap = cv2.VideoCapture(0)

def trigger_alarm():
    print("Drowsiness detected! Triggering alarm...")
    pygame.mixer.init()  # Initialize the pygame mixer
    pygame.mixer.music.load("alarm_sound.mp3")  # Load the alarm sound file
    pygame.mixer.music.play()  # Play the sound

def detect_drowsiness():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame (resize, normalize, flatten)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        resized = cv2.resize(gray, (64, 64))  # Resize to match input size
        normalized = resized / 255.0  # Normalize pixel values
        flattened = normalized.flatten().reshape(1, -1)  # Flatten the image

        # Predict drowsiness (0: awake, 1: drowsy)
        prediction = svm_model.predict(flattened)

        # Display the result on the frame
        if prediction == 1:
            cv2.putText(frame, "drowsy", (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            trigger_alarm()
        else:
            cv2.putText(frame, "Awake", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)



            # Show the frame with prediction
        cv2.imshow('Driver Drowsiness Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Start the drowsiness detection
detect_drowsiness()

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
