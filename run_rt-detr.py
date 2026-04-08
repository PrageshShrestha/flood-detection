from rfdetr import RFDETRMedium
import cv2
from PIL import Image
import supervision as sv

# === LOAD FROM YOUR LOCAL FILE ===
model_path = "/home/pragesh-shrestha/Desktop/binayak_sir/rf-detr-medium.pth"

model = RFDETRMedium(
    pretrain_weights=model_path,   # ← This is the key
    #device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    device="cpu"
)

print("✅ RF-DETR Medium loaded from local file!")

# Optimize for video inference
model.optimize_for_inference()

# === Video Inference ===
cap = cv2.VideoCapture("test.mp4")   # Change to 0 for webcam

box_annotator = sv.BoxAnnotator(thickness=3)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.7)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to PIL Image (required by RF-DETR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # Inference
    detections = model.predict(pil_image, threshold=0.25)

    # Optional: Filter only humans (class 0 = person)
    # detections = detections[detections.class_id == 0]

    # Annotate
    annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
    print(detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

    cv2.imshow("RF-DETR Medium - Video", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()