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

# Load pre-trained face and eye detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Alarm control
alarm_playing = False
geolocator = Nominatim(user_agent="drowsiness_detection")

# Email settings
EMAIL = "hk606336@gmail.com"
PASSWORD = "jcsd ablt qeol njuo"
RECIPIENT = "rawatshivam3691@gmail.com"

# Helper functions
def fetch_location():
    try:
        location = geolocator.geocode("Dehradun")  # Replace with specific city/region.
        if location:
            return f"Latitude: {location.latitude}, Longitude: {location.longitude}, Address: {location.address}"
        return "Location unavailable"
    except Exception as e:
        return f"Error fetching location: {e}"

def save_location(location):
    with open("drowsiness_locations.txt", "a") as file:
        file.write(f"{time.ctime()}: {location}\n")

def send_email(location):
    try:
        message = MIMEText(f"Drowsiness detected! Location:\n\n{location}")
        message['Subject'] = "Drowsiness Alert"
        message['From'] = EMAIL
        message['To'] = RECIPIENT
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL, PASSWORD)
            server.sendmail(EMAIL, RECIPIENT, message.as_string())
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

def trigger_alarm():
    global alarm_playing
    if not alarm_playing:
        alarm_playing = True
        pygame.mixer.init()
        pygame.mixer.music.load("alarm_sound.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pass
        alarm_playing = False

# Main drowsiness detection loop
def detect_drowsiness():
    global alarm_playing
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(face_roi)

            drowsy_flag = False
            for (ex, ey, ew, eh) in eyes:
                eye_roi = face_roi[ey:ey+eh, ex:ex+ew]

                # Preprocess the eye ROI
                resized = cv2.resize(eye_roi, (64, 64))
                normalized = resized / 255.0
                flattened = normalized.flatten().reshape(1, -1)

                # Predict using the SVM model
                prediction = svm_model.predict(flattened)[0]
                if prediction == 1:
                    drowsy_flag = True
                    break

            if drowsy_flag:
                cv2.putText(frame, "Drowsy", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                if not alarm_playing:
                    threading.Thread(target=trigger_alarm).start()
                    location = fetch_location()
                    save_location(location)
                    threading.Thread(target=send_email, args=(location,)).start()
            else:
                cv2.putText(frame, "Awake", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                if alarm_playing:
                    pygame.mixer.music.stop()
                    alarm_playing = False

        cv2.imshow('Driver Drowsiness Detection', frame)

        # Exit loop on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Run detection
detect_drowsiness()

# Release resources
cap.release()
cv2.destroyAllWindows()
