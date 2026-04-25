import torch
import cv2
import numpy as np
from pathlib import Path
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from collections import defaultdict
import time

# ===========================================================================
# CONFIGURATION
# ===========================================================================
# COCO classes that map to your competition classes
COCO_CLASSES = {
    0: 'person',           # -> Human
    2: 'car',             # -> GroundVehicle
    3: 'motorcycle',      # -> GroundVehicle
    5: 'bus',             # -> GroundVehicle
    7: 'truck',           # -> GroundVehicle
    4: 'airplane',        # -> Aircraft
    8: 'ship',            # -> Ship
    9: 'helicopter',      # -> Helicopter
    56: 'chair',          # -> Obstacle (maybe)
    72: 'refrigerator',   # -> Obstacle
    73: 'book',           # -> Obstacle
}

# Mapping COCO class IDs to competition classes
COCO_TO_COMP = {
    'person': 'Human',
    'car': 'GroundVehicle',
    'motorcycle': 'GroundVehicle',
    'bus': 'GroundVehicle',
    'truck': 'GroundVehicle',
    'airplane': 'Aircraft',
    'ship': 'Ship',
    'helicopter': 'Helicopter',
    'chair': 'Obstacle',
    'refrigerator': 'Obstacle',
    'book': 'Obstacle',
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCORE_THRESH = 0.3  # COCO model is confident, can use higher threshold

# ===========================================================================
# LOAD BASE MODEL (COCO-pretrained)
# ===========================================================================
def load_base_model():
    """Load the original COCO-pretrained RT-DETR model"""
    print("Loading COCO-pretrained RT-DETR model...")
    
    # Load model with COCO weights (91 classes)
    model = RTDetrForObjectDetection.from_pretrained(
        "PekingU/rtdetr_r101vd",
        local_files_only=False,  # Download from HuggingFace
    )
    
    # Load processor
    processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r101vd")
    
    model = model.to(DEVICE)
    model.eval()
    
    print(f"✅ Base model loaded on {DEVICE}")
    print(f"   Model has {model.config.num_labels} COCO classes")
    
    return model, processor

# ===========================================================================
# NMS FUNCTION
# ===========================================================================
def nms(boxes, scores, iou_threshold=0.5):
    """Non-Maximum Suppression"""
    if len(boxes) == 0:
        return []
    
    boxes_xyxy = boxes.copy()
    indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
            
        current_box = boxes_xyxy[current]
        rest_boxes = boxes_xyxy[indices[1:]]
        
        # Calculate IoU
        x1 = np.maximum(current_box[0], rest_boxes[:, 0])
        y1 = np.maximum(current_box[1], rest_boxes[:, 1])
        x2 = np.minimum(current_box[2], rest_boxes[:, 2])
        y2 = np.minimum(current_box[3], rest_boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        area2 = (rest_boxes[:, 2] - rest_boxes[:, 0]) * (rest_boxes[:, 3] - rest_boxes[:, 1])
        iou = intersection / (area1 + area2 - intersection + 1e-6)
        
        indices = indices[1:][iou <= iou_threshold]
    
    return keep

# ===========================================================================
# DETECT FRAME WITH COCO MODEL
# ===========================================================================
def detect_frame_coco(frame, model, processor, score_thresh=0.3, iou_thresh=0.5):
    """Detect objects using COCO-pretrained model and map to competition classes"""
    
    height, width = frame.shape[:2]
    
    # Preprocess
    inputs = processor(images=frame, return_tensors="pt").to(DEVICE)
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Decode predictions (COCO: 80 classes)
    logits = outputs.logits[0].sigmoid()  # (num_queries, 80)
    boxes = outputs.pred_boxes[0]         # (num_queries, 4) cxcywh normalized
    
    # Get predictions
    scores, labels = logits.max(dim=-1)
    keep_scores = scores > score_thresh
    
    if not keep_scores.any():
        return []
    
    # Convert to pixel coordinates
    cx, cy, w, h = boxes[keep_scores].unbind(-1)
    x1 = (cx - w/2) * width
    y1 = (cy - h/2) * height
    x2 = (cx + w/2) * width
    y2 = (cy + h/2) * height
    
    # Collect detections
    detections = []
    for i in range(len(scores[keep_scores])):
        coco_class_id = labels[i].item()
        coco_class_name = model.config.id2label.get(coco_class_id, 'unknown')
        
        # Map to competition class if possible
        comp_class = COCO_TO_COMP.get(coco_class_name, None)
        
        if comp_class:  # Only keep mappable classes
            detections.append({
                'class': comp_class,
                'coco_class': coco_class_name,
                'score': scores[i].item(),
                'box': [x1[i].item(), y1[i].item(), x2[i].item(), y2[i].item()]
            })
    
    # Apply NMS per competition class
    final_detections = []
    unique_classes = set(d['class'] for d in detections)
    
    for comp_class in unique_classes:
        class_dets = [d for d in detections if d['class'] == comp_class]
        if not class_dets:
            continue
        
        boxes_class = np.array([d['box'] for d in class_dets])
        scores_class = np.array([d['score'] for d in class_dets])
        
        keep_indices = nms(boxes_class, scores_class, iou_thresh)
        
        for idx in keep_indices:
            final_detections.append(class_dets[idx])
    
    return final_detections

# ===========================================================================
# DRAW DETECTIONS
# ===========================================================================
def draw_detections(frame, detections):
    """Draw bounding boxes with class names"""
    
    # Color mapping for competition classes
    COLORS = {
        'Aircraft': (255, 0, 0),
        'Human': (0, 255, 0),
        'GroundVehicle': (0, 255, 255),
        'Drone': (255, 0, 255),
        'Ship': (0, 165, 255),
        'Obstacle': (128, 0, 128),
        'Helicopter': (255, 255, 0),
    }
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det['box'])
        class_name = det['class']
        score = det['score']
        
        color = COLORS.get(class_name, (255, 255, 255))
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}: {score:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

# ===========================================================================
# VIDEO PROCESSING
# ===========================================================================
def process_video_base(video_path, output_path, model, processor, display=True):
    """Process video using base COCO model"""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Statistics
    frame_count = 0
    detection_counts = defaultdict(int)
    processing_times = []
    
    print(f"Processing video: {total_frames} frames")
    print(f"Resolution: {width}x{height}, FPS: {fps}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process frame
        start_time = time.time()
        
        # For better performance, optionally resize frame
        # frame_small = cv2.resize(frame, (960, 540))
        # detections = detect_frame_coco(frame_small, model, processor, SCORE_THRESH)
        # # Scale boxes back
        # scale_x = width / 960
        # scale_y = height / 540
        # for det in detections:
        #     det['box'] = [det['box'][0]*scale_x, det['box'][1]*scale_y,
        #                  det['box'][2]*scale_x, det['box'][3]*scale_y]
        
        detections = detect_frame_coco(frame, model, processor, SCORE_THRESH)
        
        inference_time = time.time() - start_time
        processing_times.append(inference_time)
        
        # Update statistics
        for det in detections:
            detection_counts[det['class']] += 1
        
        # Draw detections
        annotated_frame = draw_detections(frame.copy(), detections)
        
        # Add FPS overlay
        avg_fps = 1.0 / np.mean(processing_times[-30:]) if processing_times else 0
        cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Objects: {len(detections)}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Progress
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}/{total_frames} | "
                  f"Detections: {len(detections)} | "
                  f"FPS: {avg_fps:.1f}")
        
        # Write and display
        out.write(annotated_frame)
        
        if display:
            cv2.imshow('RT-DETR Base Model Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Cleanup
    cap.release()
    out.release()
    if display:
        cv2.destroyAllWindows()
    
    # Print summary
    print("\n" + "="*50)
    print("VIDEO PROCESSING COMPLETE")
    print(f"Total frames: {frame_count}")
    print(f"Average FPS: {1.0/np.mean(processing_times):.1f}")
    print(f"Output saved to: {output_path}")
    print("\nDetection summary:")
    for class_name, count in sorted(detection_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count} detections")
    print("="*50)

# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    # Load base COCO model (no checkpoint needed!)
    model, processor = load_base_model()
    
    # Process video
    process_video_base(
        video_path="test2.mp4",  # Your video file
        output_path="output_base_model.mp4",
        model=model,
        processor=processor,
        display=True
    )
