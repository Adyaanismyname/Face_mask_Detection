import cv2
import tensorflow as tf
import numpy as np
import os
import time

# Create a directory to save the face images if it doesn't exist
save_dir = "saved_faces"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Load the model once, outside the loop
model = tf.keras.models.load_model('mask_detection_model.keras')
# Start video capture (0 is the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the camera
    ret, frame = cap.read()
    # Convert the frame to grayscale (Haar Cascade works on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Collect the faces for batch prediction
    cropped_faces_original = []  # Store original cropped faces for saving
    
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face (for visual confirmation)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Crop the face from the frame
        cropped_face = frame[y:y + h, x:x + w]
        # Resize the cropped face to the input size expected by the model
        resized_face = cv2.resize(cropped_face, (224, 224))
        resized_face = resized_face.reshape(1,224,224,3)
        # Normalize the cropped face
        normalized_face = resized_face / 255.0

        cv2.imshow("Face", resized_face[0])


        predictions = model.predict(normalized_face)
        label = np.argmax(predictions)
        print(label)
        # Determine the label text
        if label == 2:
            text = "With Mask"
        elif label == 1:
            text = "Without Mask"
        else:
            text = "Unknown"

        # Show label text on the video frame
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 255, 0) if label == 2 else (0, 0, 255), 2)
    
     
    # Show the original frame with rectangles around detected faces
    cv2.imshow("Frame", frame)
    
    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    
    # Break the loop if the user presses 'q'
    if key == ord('q'):
        break
    
# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()