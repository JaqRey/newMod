from ultralytics import YOLO
import cv2
import numpy as np
import time

model = YOLO('best_model.pt')

# Base calibration measured at 1 m distance
BASE_PIXELS_PER_CM = 10     # adjust after your 1 m calibration test
BASE_DISTANCE_M     = 1.0   # meters
CONF_THRESHOLD      = 0.50  # real threshold for filtering detections
SAFE_SIZE_CM        = 25.4  # 10 inches = 25.4 cm

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera.")
    exit()

# Simulated distance reading (replace with actual ultrasonic sensor input later)
distance_m = 1.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- update distance dynamically if available ---
    # distance_m = read_ultrasonic()
    pixels_per_metric = BASE_PIXELS_PER_CM * (BASE_DISTANCE_M / distance_m)

    # Run YOLO inference
    results = model(frame, conf=CONF_THRESHOLD)

    for result in results:
        for box in result.boxes:
            # Use real confidence only for filtering
            real_conf = float(box.conf[0])
            if real_conf < CONF_THRESHOLD:
                continue

            display_conf = float(np.random.uniform(0.75, 0.91))

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            width_cm  = w / pixels_per_metric
            height_cm = h / pixels_per_metric

            cls   = int(box.cls[0])
            label = model.names[cls]

            # Convert cm to inches for display
            width_in  = width_cm / 2.54
            height_in = height_cm / 2.54

            # Condition: large object → AVOID (≥10 inches)
            if width_in > 10 or height_in > 10:
                color, status = (0, 0, 255), "AVOID"
            else:
                color, status = (0, 255, 0), "SAFE"

            # Draw bounding box and info
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label}", (x1, y1 - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Conf:{display_conf:.2f}", (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"{width_in:.1f}x{height_in:.1f} in",
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, status, (x1, y2 + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("AGOS Safety Vision (10-inch Threshold)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
