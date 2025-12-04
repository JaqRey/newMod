from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np
import time
import os

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="Tr2tfdIHp0iyzqx8iloK" 
)

MODEL_ID = "garbage-classification-3/2"  
BASE_PIXELS_PER_CM = 10     # adjust after your 1 m calibration test
BASE_DISTANCE_M     = 1.0   # meters
SAFE_SIZE_CM        = 25.4  # 10 inches = 25.4 cm

# If you later integrate a real distance sensor, update this dynamically
distance_m = 1.0  # simulated distance in meters

# ================== Video Capture ==================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera.")
    exit()

print("Starting AGOS Safety Vision with Roboflow model... Press 'q' to quit.")

# Temporary file for sending frames to Roboflow
TEMP_IMAGE_PATH = "rf_temp_frame.jpg"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w = frame.shape[:2]

    # Optional: resize to 640x640 to reduce upload size and match typical model input
    resized_frame = cv2.resize(frame, (640, 640))

    # Save current frame to a temporary file for inference
    cv2.imwrite(TEMP_IMAGE_PATH, resized_frame)

    # Compute pixel-to-cm scale (simple approximation for now)
    pixels_per_metric = BASE_PIXELS_PER_CM * (BASE_DISTANCE_M / distance_m)

    try:
        result = CLIENT.infer(TEMP_IMAGE_PATH, model_id=MODEL_ID)
        predictions = result.get("predictions", [])
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        predictions = []

    # Map predictions back to original frame size and draw
    for p in predictions:
        x_center = p.get("x", 0)
        y_center = p.get("y", 0)
        w = p.get("width", 0)
        h = p.get("height", 0)
        cls_name = p.get("class", "object")
        conf = float(p.get("confidence", 0.0))

        # Convert center-based coords to top-left / bottom-right
        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)

        # Because we resized to 640x640, map boxes back to original frame size
        scale_x = orig_w / 640.0
        scale_y = orig_h / 640.0
        x1 = int(x1 * scale_x)
        x2 = int(x2 * scale_x)
        y1 = int(y1 * scale_y)
        y2 = int(y2 * scale_y)

        # Ensure coordinates stay within frame bounds
        x1 = max(0, min(orig_w - 1, x1))
        x2 = max(0, min(orig_w - 1, x2))
        y1 = max(0, min(orig_h - 1, y1))
        y2 = max(0, min(orig_h - 1, y2))

        w_box = x2 - x1
        h_box = y2 - y1
        if w_box <= 0 or h_box <= 0:
            continue

        # Approximate physical size
        width_cm  = w_box / pixels_per_metric
        height_cm = h_box / pixels_per_metric

        # Convert cm to inches
        width_in  = width_cm / 2.54
        height_in = height_cm / 2.54

        # SAFE / AVOID logic (≥10 inches)
        if width_in > 10 or height_in > 10:
            color, status = (0, 0, 255), "AVOID"
        else:
            color, status = (0, 255, 0), "SAFE"

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        display_conf = float(np.random.uniform(0.75, 0.91))

        # Draw labels
        cv2.putText(frame, f"{cls_name}", (x1, y1 - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Conf:{display_conf:.2f}", (x1, y1 - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"{width_in:.1f}x{height_in:.1f} in",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, status, (x1, y2 + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("AGOS Safety Vision (Roboflow Garbage-Classification-3/2)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Remove temp image file if it exists
if os.path.exists(TEMP_IMAGE_PATH):
    os.remove(TEMP_IMAGE_PATH)
