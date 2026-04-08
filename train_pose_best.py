from ultralytics import YOLO

# Load your existing best.pt model
model = YOLO("best.pt") 

# Train on merged C2A dataset
results = model.train(
    data="data.yaml", 
    epochs=50,  # Fewer epochs since we're fine-tuning
    imgsz=640,
    batch=16,
    amp=False,
    name="pose_best_finetuned"
)

print("Training completed!")
print(f"Model saved to: runs/detect/pose_best_finetuned/best.pt")
