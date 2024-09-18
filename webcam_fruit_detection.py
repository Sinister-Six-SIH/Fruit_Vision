import cv2
import numpy as np
import joblib

# Load the trained model
model = joblib.load('fruit_classifier_model.pkl')

def resize_image(image, target_size=(64, 64)):  # Adjust this based on the size used in training
    return cv2.resize(image, target_size)

def extract_color_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def apply_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges.flatten()

def extract_features(image):
    color_features = extract_color_histogram(image)
    edge_features = apply_edge_detection(image)
    return np.concatenate((color_features, edge_features))

# Access webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting...")
        break
    
    resized_frame = resize_image(frame)
    # Extract features from the current frame
    features = extract_features(resized_frame).reshape(1, -1)  # Reshape to match the expected input for the model
    print(f"Feature size for current frame: {features.size}")

    if features.size == 10512:  # Check if feature size matches the model's expected input
        # Make predictions
        prediction = model.predict(features)

        # Display the prediction
        label = "Apple" if prediction[0] == 0 else "Banana"
        cv2.putText(frame, f"Prediction: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        print(f"Feature size mismatch. Expected 10512 but got {features.size}")

    # Show the frame to the user
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
