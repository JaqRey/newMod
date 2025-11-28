from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Calibration value (pixels per centimeter)
pixels_per_metric = 17  # adjust this based on your calibration setup

# Confidence threshold (at least 85%)
CONF_THRESHOLD = 0.85

# Use external webcam (try 0, 1, or 2 depending on your system)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open external webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not captured. Check camera connection.")
        break

    # Run YOLO detection with confidence threshold
    results = model(frame, conf=CONF_THRESHOLD)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue  # Skip detections below 85% confidence

            # Extract coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1

            # Convert to centimeters
            width_cm = w / pixels_per_metric
            height_cm = h / pixels_per_metric

            # Class and label
            cls = int(box.cls[0])
            label = model.names[cls]

            # Condition: large object → AVOID
            if width_cm > 10 or height_cm > 10:
                color = (0, 0, 255)  # Red
                status = "AVOID"
            else:
                color = (0, 255, 0)  # Green
                status = "SAFE"

            # Draw bounding box and text
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label}", (x1, y1 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Conf: {conf:.2f}", (x1, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"{width_cm:.1f}x{height_cm:.1f} cm", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, status, (x1, y2 + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("YOLOv8 + Size Measurement (≥85% Confidence)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
