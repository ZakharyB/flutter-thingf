import cv2
import numpy as np
from fer import FER
from ultralytics import YOLO

# Initialize YOLO model for face detection
yolo_model = YOLO('yolov8n-face.pt')

# Initialize FER for emotion detection
emotion_detector = FER(mtcnn=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces using YOLO
    results = yolo_model(frame)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Extract face region
            face = frame[y1:y2, x1:x2]

            # Detect emotion using FER
            emotions = emotion_detector.detect_emotions(face)
            if emotions:
                emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add emotion label
                cv2.putText(frame, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
