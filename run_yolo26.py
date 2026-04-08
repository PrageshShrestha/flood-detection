from ultralytics import YOLO
import cv2

# Load YOLO26x model (auto-downloads ~113 MB on first run)
model = YOLO("yolo26x.pt")   # Detection model

# Video inference settings
results = model.predict(
    source="test.mp4",          # Change to your video path or 0 for webcam
    show=True,                  # Display live window
    conf=0.25,                  # Confidence threshold (0.25 is good balance)
    iou=0.45,                   # NMS IoU threshold
    imgsz=640,                  # Image size (640 is standard, 1280 for higher accuracy)
    stream=True,                # Memory efficient for videos
    device="0",                 # Use "0" for GPU, "cpu" for CPU
    verbose=False               # Clean output
)

# Process the video
for r in results:
    # You can access detections here if needed
    boxes = r.boxes  # Bounding boxes
    # classes = r.boxes.cls
    # confs = r.boxes.conf

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()