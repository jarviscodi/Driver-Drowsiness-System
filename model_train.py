import os
import cv2
import joblib
import numpy as np
from sklearn.svm import SVC

from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def preprocess_images_with_hog(folder_path):
    features = []
    labels = []
    for label, class_name in enumerate(['awake', 'drowsy']):
        class_path = os.path.join(folder_path, class_name)
        for filename in os.listdir(class_path):
            image_path = os.path.join(class_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
            if image is not None:
                image = cv2.resize(image, (64, 64))  # Resize images
                # Extract HOG features
                hog_features = hog(
                    image,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    block_norm='L2-Hys',
                    transform_sqrt=True,
                )
                features.append(hog_features)
                labels.append(label)  # 0 for awake, 1 for drowsy
    return np.array(features), np.array(labels)

# Load and preprocess data
X, y = preprocess_images_with_hog('dataset')

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SVM model
svm_model = SVC(kernel='linear')  # Linear kernel for simplicity

# Train the model
svm_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = svm_model.predict(X_val)

# Evaluate the model's performance
accuracy = accuracy_score(y_val, y_pred)
print(f"SVM Model Accuracy (Validation): {accuracy * 100:.2f}%")


X_test, y_test = preprocess_images_with_hog('dataset')
# Evaluate the model on test data
y_test_pred = svm_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"SVM Model Test Accuracy: {test_accuracy * 100:.2f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
print(f"confusion Matrix: {conf_matrix}")
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Awake', 'Drowsy'], yticklabels=['Awake', 'Drowsy'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save the trained model to a file
joblib.dump(svm_model, 'drowsiness_detection_svm_model_with_hog.pkl')
