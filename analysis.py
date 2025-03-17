import os
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Function to load and preprocess images

def preprocess_images(folder_path, use_hog=False):
    images = []
    labels = []
    for label, class_name in enumerate(['awake', 'drowsy']):
        class_path = os.path.join(folder_path, class_name)
        for filename in os.listdir(class_path):
            image_path = os.path.join(class_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
            if image is not None:
                image = cv2.resize(image, (64, 64))  # Resize images
                if use_hog:
                    # Extract HOG features
                    fd, _ = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
                    images.append(fd)  # HOG features
                else:
                    images.append(image.flatten())  # Raw pixel features
                labels.append(label)
    return np.array(images), np.array(labels)

# Load and preprocess images
X_raw, y = preprocess_images('dataset', use_hog=False)  # Raw pixel features
X_hog, _ = preprocess_images('dataset', use_hog=True)  # HOG features

# Normalize raw pixel features to range [0, 1]
X_raw = X_raw / 255.0

# Split the data for both raw and HOG features
X_raw_train, X_raw_val, y_train, y_val = train_test_split(X_raw, y, test_size=0.2, random_state=42)
X_hog_train, X_hog_val, _, _ = train_test_split(X_hog, y, test_size=0.2, random_state=42)

# Initialize SVM models for both approaches
svm_raw = SVC(kernel='linear')
svm_hog = SVC(kernel='linear')

# Train the models
svm_raw.fit(X_raw_train, y_train)
svm_hog.fit(X_hog_train, y_train)

# Make predictions on validation sets
y_raw_pred = svm_raw.predict(X_raw_val)
y_hog_pred = svm_hog.predict(X_hog_val)

# Evaluate both models
accuracy_raw = accuracy_score(y_val, y_raw_pred)
accuracy_hog = accuracy_score(y_val, y_hog_pred)
print(f"SVM Model Accuracy with Raw Features: {accuracy_raw * 100:.2f}%")
print(f"SVM Model Accuracy with HOG Features: {accuracy_hog * 100:.2f}%")

# Generate confusion matrices
cm_raw = confusion_matrix(y_val, y_raw_pred)
cm_hog = confusion_matrix(y_val, y_hog_pred)

# Display confusion matrices
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ConfusionMatrixDisplay(cm_raw, display_labels=['Awake', 'Drowsy']).plot(ax=ax[0], cmap='Blues', colorbar=False)
ax[0].set_title('Confusion Matrix - Raw Features')

ConfusionMatrixDisplay(cm_hog, display_labels=['Awake', 'Drowsy']).plot(ax=ax[1], cmap='Greens', colorbar=False)
ax[1].set_title('Confusion Matrix - HOG Features')
# Plotting the accuracies for comparative analysis
fig, ax = plt.subplots(figsize=(8, 6))

# Accuracy values
models = ['Raw Features', 'HOG Features']
accuracies = [accuracy_raw * 100, accuracy_hog * 100]

# Bar chart
ax.bar(models, accuracies, color=['blue', 'green'], alpha=0.7)
ax.set_title('Comparative Analysis of SVM Models', fontsize=16)
ax.set_ylabel('Accuracy (%)', fontsize=14)
ax.set_xlabel('Feature Type', fontsize=14)
ax.set_ylim(0, 100)
ax.bar_label(ax.containers[0], fmt='%.2f%%', label_type='edge', fontsize=12)

# Display the bar chart
plt.tight_layout()
plt.show()


plt.tight_layout()
plt.show()
