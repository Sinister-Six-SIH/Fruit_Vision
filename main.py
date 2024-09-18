import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load images and corresponding labels
def load_images_and_labels(folders):
    images = []
    labels = []
    
    for label, folder in enumerate(folders):
        for subfolder in os.listdir(folder):
            subfolder_path = os.path.join(folder, subfolder)
            # Check if it's a directory
            if os.path.isdir(subfolder_path):
                for filename in os.listdir(subfolder_path):
                    img_path = os.path.join(subfolder_path, filename)
                    print(f"Trying to load image from path: {img_path}")  # Print the path for debugging
                    image = cv2.imread(img_path)
                    if image is not None:
                        images.append(image)
                        labels.append(label)
                    else:
                        print(f"Error loading image at {img_path}")
    return images, labels

# Extract color histogram
def extract_color_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Apply edge detection
def apply_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges.flatten()

# Extract features
def extract_features(image):
    color_features = extract_color_histogram(image)
    edge_features = apply_edge_detection(image)
    return np.concatenate((color_features, edge_features))

# Load and extract features
def load_and_extract_features(folders):
    images, labels = load_images_and_labels(folders)
    features = [extract_features(img) for img in images]
    return np.array(features), np.array(labels)

# Define folders for each class
folders = ['data/0', 'data/1']  # Folder paths for different classes

# Load features and labels
X, y = load_and_extract_features(folders)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SVM classifier
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print evaluation metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import joblib

# Save the model to a file
joblib.dump(model, 'fruit_classifier_model.pkl')



