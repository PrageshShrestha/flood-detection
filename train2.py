from ultralytics import YOLO

# Load the pose estimation model
model = YOLO("yolo26x.pt") 

# Train the model
results = model.train(data="dataset.yaml", epochs=100, imgsz=640,batch=6)
