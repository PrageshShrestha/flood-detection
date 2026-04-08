from ultralytics import YOLO

# Load the Nano model
model = YOLO("yolo26n.pt") 

# Train the model
results = model.train(data="data.yaml", epochs=100, imgsz=640)
