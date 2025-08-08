import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time

# --- Initialize ---
model = YOLO('yolov8n.pt') 
tracker = DeepSort(max_age=30)

# --- Video Source ---
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or "video.mp4" for video file

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Run YOLO object detection ---
    results = model(frame, stream=True)
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            if conf < 0.5:
                continue

            if class_name == "toothbrush":
                continue

            bbox = [x1, y1, x2 - x1, y2 - y1]  # (x, y, width, height)
            detections.append((bbox, conf, class_name))

    # --- Track objects ---
    tracks = tracker.update_tracks(detections, frame=frame)

    # --- Draw bounding boxes and tracking IDs ---
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        class_name = track.get_det_class() or "object"

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label with class and ID
        label = f"{class_name} ID: {track_id}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Show frame
    cv2.imshow('Object Detection & Tracking', frame)

    # Quit on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

    if time.time() - start_time > 300:
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
