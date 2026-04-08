from ultralytics import YOLO

# Load the pose estimation model
model = YOLO("best.pt") 

# Train the model
results = model.train(data="data2.yaml", epochs=100, imgsz=640)
