import cv2
import joblib
import numpy as np
import pygame
import threading
from geopy.geocoders import Nominatim
import smtplib
from email.mime.text import MIMEText
import time

# Load the trained model
svm_model = joblib.load('drowsiness_detection_svm_model.pkl')

# Initialize the webcam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Alarm control
alarm_playing = False

# Initialize location fetcher
geolocator = Nominatim(user_agent="drowsiness_detection")

# Email settings
EMAIL = "hk606336@gmail.com"
PASSWORD = "13@10@2001"
RECIPIENT = "himanshukumar07000@gmail.com"


def fetch_location():

    try:
        location = geolocator.geocode("Dehradun")  # Replace "Your City" with a specific city or region for accuracy.
        if location:
            return f"Latitude: {location.latitude}, Longitude: {location.longitude}, Address: {location.address}"
        return "Location unavailable"
    except Exception as e:
        return f"Error fetching location: {e}"


def save_location(location):
    """Saves the detected location to a file."""
    with open("drowsiness_locations.txt", "a") as file:
        file.write(f"{time.ctime()}: {location}\n")


def send_email(location):
    """Sends an email with the detected location."""
    try:
        message = MIMEText(f"Drowsiness detected! Here is the location:\n\n{location}")
        message['Subject'] = "Drowsiness Alert"
        message['From'] = EMAIL
        message['To'] = RECIPIENT

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL, PASSWORD)
            server.sendmail(EMAIL, RECIPIENT, message.as_string())
        print("Alert email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")


def trigger_alarm():
    """Plays an alarm sound."""
    global alarm_playing
    if not alarm_playing:
        alarm_playing = True
        pygame.mixer.init()
        pygame.mixer.music.load("alarm_sound.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pass
        alarm_playing = False


def detect_drowsiness():
    global alarm_playing
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors =5)
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]

            # Detect eyes within the face region
            eyes = eye_cascade.detectMultiScale(face_roi)
            for (ex, ey, ew, eh) in eyes:
                eye_roi = face_roi[ey:ey + eh, ex:ex + ew]

        resized = cv2.resize(gray, (64, 64))
        normalized = resized / 255.0
        flattened = normalized.flatten().reshape(1, -1)

        # Predict drowsiness
        prediction = svm_model.predict(flattened)

        if prediction == 1:  # Drowsy
            cv2.putText(frame, "Drowsy", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            if not alarm_playing:
                threading.Thread(target=trigger_alarm).start()
                location = fetch_location()
                save_location(location)
                send_email(location)
        else:  # Awake
            cv2.putText(frame, "Awake", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if alarm_playing:
                pygame.mixer.music.stop()
                alarm_playing = False

        # Display the frame
        cv2.imshow('Driver Drowsiness Detection', frame)

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Start detection
detect_drowsiness()

# Release resources
cap.release()
cv2.destroyAllWindows()
