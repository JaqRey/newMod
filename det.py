from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np
import os

# ================== Roboflow Client Config ==================
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="Tr2tfdIHp0iyzqx8iloK"  # <-- put your Roboflow API key here
)

MODEL_ID = "garbage-classification-3/2"  # <-- from Roboflow Deploy page

# ================== AGOS Size & Safety Config ==================
BASE_PIXELS_PER_CM = 10      # calibration at 1 m
BASE_DISTANCE_M     = 1.0    # meters
SAFE_SIZE_CM        = 25.4   # 10 inches = 25.4 cm
distance_m          = 1.0    # simulated distance (replace with real sensor later)

# Speed tweaks
RESIZE_W, RESIZE_H = 416, 416   # smaller than 640x640 for faster upload/inference
FRAME_SKIP          = 3         # call API every 3rd frame

# ================== Video Capture ==================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera.")
    exit()

print("Starting AGOS Safety Vision with Roboflow model... Press 'q' to quit.")

TEMP_IMAGE_PATH = "rf_temp_frame.jpg"
frame_count = 0
last_predictions = []   # reuse last predictions between API calls

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    orig_h, orig_w = frame.shape[:2]

    # Only send every Nth frame to Roboflow to reduce lag
    if frame_count % FRAME_SKIP == 0:
        # Resize the frame for faster upload / inference
        resized_frame = cv2.resize(frame, (RESIZE_W, RESIZE_H))

        # Save to temp file for the SDK
        cv2.imwrite(TEMP_IMAGE_PATH, resized_frame)

        try:
            result = CLIENT.infer(TEMP_IMAGE_PATH, model_id=MODEL_ID)
            last_predictions = result.get("predictions", [])
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            last_predictions = []

    # Use the latest predictions (from the most recent API call)
    predictions = last_predictions
    pixels_per_metric = BASE_PIXELS_PER_CM * (BASE_DISTANCE_M / distance_m)

    # Draw predictions on the current frame
    for p in predictions:
        # Roboflow returns center x,y and width/height in pixels (relative to the resized image)
        x_center = p.get("x", 0)
        y_center = p.get("y", 0)
        w_box    = p.get("width", 0)
        h_box    = p.get("height", 0)
        cls_name = p.get("class", "object")
        conf     = float(p.get("confidence", 0.0))

        # Convert center-based coords (on RESIZE_W x RESIZE_H) to x1,y1,x2,y2
        x1 = int(x_center - w_box / 2)
        y1 = int(y_center - h_box / 2)
        x2 = int(x_center + w_box / 2)
        y2 = int(y_center + h_box / 2)

        # Map back to original frame size
        scale_x = orig_w / float(RESIZE_W)
        scale_y = orig_h / float(RESIZE_H)
        x1 = int(x1 * scale_x)
        x2 = int(x2 * scale_x)
        y1 = int(y1 * scale_y)
        y2 = int(y2 * scale_y)

        # Clamp to frame bounds
        x1 = max(0, min(orig_w - 1, x1))
        x2 = max(0, min(orig_w - 1, x2))
        y1 = max(0, min(orig_h - 1, y1))
        y2 = max(0, min(orig_h - 1, y2))

        w_box_orig = x2 - x1
        h_box_orig = y2 - y1
        if w_box_orig <= 0 or h_box_orig <= 0:
            continue

        # Approximate physical size
        width_cm  = w_box_orig / pixels_per_metric
        height_cm = h_box_orig / pixels_per_metric

        # Convert to inches
        width_in  = width_cm / 2.54
        height_in = height_cm / 2.54

        # SAFE / AVOID logic (≥ 10 inches)
        if width_in > 10 or height_in > 10:
            color, status = (0, 0, 255), "AVOID"
        else:
            color, status = (0, 255, 0), "SAFE"

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Show REAL confidence from Roboflow (0–1)
        display_conf = conf

        # Labels
        cv2.putText(frame, f"{cls_name}", (x1, y1 - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Conf:{display_conf:.2f}", (x1, y1 - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"{width_in:.1f}x{height_in:.1f} in",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, status, (x1, y2 + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("AGOS Safety Vision (Roboflow, Optimized)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

if os.path.exists(TEMP_IMAGE_PATH):
    os.remove(TEMP_IMAGE_PATH)
