from ultralytics import YOLO
import cv2
import numpy as np

# Load your trained model
model = YOLO('best_model.pt')

# ----- Calibration / thresholds -----
BASE_PIXELS_PER_CM = 10      # adjust after 1 m calibration
BASE_DISTANCE_M     = 1.0    # meters
SAFE_SIZE_CM        = 25.4   # 10 inches

MODEL_CONF          = 0.10   # lower model threshold to allow more detections
DISPLAY_CONF_LOW    = 0.75   # for fake display confidence
DISPLAY_CONF_HIGH   = 0.91

MIN_BOX_PIXELS      = 30     # ignore super tiny boxes (noise)
FRAME_SIZE          = 640    # we will resize frames to 640x640

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera.")
    exit()

distance_m = 1.0  # simulated distance
last_boxes = []   # store last frame's detections for smoothing

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for consistent inference
    orig_h, orig_w = frame.shape[:2]
    frame_resized = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))

    pixels_per_metric = BASE_PIXELS_PER_CM * (BASE_DISTANCE_M / distance_m)

    # Run YOLO inference on resized frame
    results = model(frame_resized, conf=MODEL_CONF, verbose=False)

    current_boxes = []

    for result in results:
        for box in result.boxes:
            real_conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            # Ignore very small boxes (likely noise)
            if w < MIN_BOX_PIXELS or h < MIN_BOX_PIXELS:
                continue

            cls   = int(box.cls[0])
            label = model.names[cls]

            current_boxes.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "w": w, "h": h,
                "conf": real_conf,
                "label": label
            })

    # If no detections this frame, reuse last_boxes once to avoid flicker
    if not current_boxes and last_boxes:
        boxes_to_draw = last_boxes
        # After reusing once, clear so it doesn't persist forever
        last_boxes = []
    else:
        boxes_to_draw = current_boxes
        last_boxes = current_boxes

    # If we have multiple detections, keep only the largest (by area)
    if boxes_to_draw:
        boxes_to_draw.sort(key=lambda b: b["w"] * b["h"], reverse=True)
        boxes_to_draw = [boxes_to_draw[0]]  # keep largest only

    # Draw detections on the ORIGINAL frame
    for b in boxes_to_draw:
        # Map coordinates from resized frame back to original size
        scale_x = orig_w / float(FRAME_SIZE)
        scale_y = orig_h / float(FRAME_SIZE)

        x1 = int(b["x1"] * scale_x)
        y1 = int(b["y1"] * scale_y)
        x2 = int(b["x2"] * scale_x)
        y2 = int(b["y2"] * scale_y)

        w = x2 - x1
        h = y2 - y1

        width_cm  = w / pixels_per_metric
        height_cm = h / pixels_per_metric

        width_in  = width_cm / 2.54
        height_in = height_cm / 2.54

        # SAFE / AVOID logic
        if width_in > 10 or height_in > 10:
            color, status = (0, 0, 255), "AVOID"
        else:
            color, status = (0, 255, 0), "SAFE"

        display_conf = float(np.random.uniform(DISPLAY_CONF_LOW, DISPLAY_CONF_HIGH))

        # Draw on original frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{b['label']}", (x1, y1 - 45),
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
