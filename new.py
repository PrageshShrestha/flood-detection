#!/usr/bin/env python3
"""
Standalone Video Inference Script for RT-DETR Airborne Object Detection

This script loads your trained RT-DETR model from .pth weights only
and performs inference on video files.

Requirements:
    pip install torch torchvision transformers opencv-python pillow numpy

Usage:
    python video_inference.py --weights best_rtdetr.pth --video test_video.mp4
    python video_inference.py --weights best_rtdetr.pth --video test_video.mp4 --output result.mp4
    python video_inference.py --weights best_rtdetr.pth --webcam 0
"""

import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import json
import time
import warnings
from collections import OrderedDict

warnings.filterwarnings("ignore")

# ===========================================================================
# CONFIGURATION
# ===========================================================================
class Config:
    # Classes (same as your training)
    CLASSES = ["Aircraft", "Human", "GroundVehicle", "Drone", "Ship", "Obstacle", "Helicopter"]
    NUM_CLASSES = len(CLASSES)
    IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
    
    # Inference settings
    CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for detection
    IOU_THRESHOLD = 0.5         # NMS IoU threshold
    IMAGE_SIZE = 800            # Input size for model (must match training)
    
    # Visualization colors (BGR format for OpenCV)
    COLORS = {
        "Aircraft": (0, 0, 255),      # Red
        "Human": (0, 255, 0),          # Green
        "GroundVehicle": (255, 0, 0),  # Blue
        "Drone": (255, 255, 0),        # Cyan
        "Ship": (255, 0, 255),         # Magenta
        "Obstacle": (0, 255, 255),     # Yellow
        "Helicopter": (128, 0, 128),   # Purple
    }
    
    # Display settings
    SHOW_FPS = True
    LINE_THICKNESS = 2
    FONT_SCALE = 0.5
    
    # Hardware
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = True  # Mixed precision for faster inference


# ===========================================================================
# RT-DETR MODEL DEFINITION (Simplified for inference)
# ===========================================================================
class RTDETRDetector:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        
    def load_model(self, weights_path: str):
        """Load the trained RT-DETR model from .pth weights only"""
        print(f"Loading model from: {weights_path}")
        print(f"Device: {self.config.DEVICE}")
        
        # Load the transformers model (this will download the architecture)
        # If you don't have internet, you need to specify a local cache
        from transformers import RTDetrForObjectDetection
        
        # First, load the base model architecture from pretrained
        # Note: This requires internet connection to download the config
        # If you're offline, you'll need to have the model cached
        try:
            # Try to load from local cache first
            self.model = RTDetrForObjectDetection.from_pretrained(
                "PekingU/rtdetr_r101vd",  # This will download if not cached
                num_labels=self.config.NUM_CLASSES,
                ignore_mismatched_sizes=True,
            )
        except:
            # Alternative: Use a smaller backbone if R101 is not available
            print("R101 backbone not available, using R18 for inference...")
            self.model = RTDetrForObjectDetection.from_pretrained(
                "PekingU/rtdetr_r18vd",
                num_labels=self.config.NUM_CLASSES,
                ignore_mismatched_sizes=True,
            )
        
        # Load your trained weights
        checkpoint = torch.load(weights_path, map_location=self.config.DEVICE)
        
        # Handle different checkpoint formats
        if "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        
        # Load weights, ignoring missing/unexpected keys
        missing_keys, unexpected_keys = self.model.load_state_dict(
            new_state_dict, strict=False
        )
        
        print(f"Loaded checkpoint with val mAP@0.5: {checkpoint.get('val_map50', 'N/A')}")
        if missing_keys:
            print(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys[:5]}...")
        
        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        
        print("Model loaded successfully!")
        
    def preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, float, Tuple[int, int]]:
        """Preprocess image for model input"""
        h, w = image.shape[:2]
        scale = self.config.IMAGE_SIZE / max(h, w)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Pad to square
        pad_w = self.config.IMAGE_SIZE - new_w
        pad_h = self.config.IMAGE_SIZE - new_h
        
        padded = cv2.copyMakeBorder(
            resized, 0, pad_h, 0, pad_w, 
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
        
        # Normalize with ImageNet stats (standard for RT-DETR)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        return img_tensor.unsqueeze(0), scale, (pad_w, pad_h)
    
    def postprocess_predictions(
        self, 
        outputs, 
        scale: float, 
        pad: Tuple[int, int],
        original_size: Tuple[int, int]
    ) -> List[Dict]:
        """Convert model outputs to detection boxes"""
        # Get predictions
        if hasattr(outputs, 'logits'):
            logits = outputs.logits[0].sigmoid()
            boxes = outputs.pred_boxes[0]
        else:
            # Handle different output formats
            logits = outputs[0][0].sigmoid() if isinstance(outputs, tuple) else outputs[0].sigmoid()
            boxes = outputs[1][0] if isinstance(outputs, tuple) else outputs[1]
        
        # Get class predictions
        scores, labels = logits.max(dim=-1)
        
        # Filter by confidence
        keep = scores > self.config.CONFIDENCE_THRESHOLD
        scores = scores[keep].cpu().numpy()
        labels = labels[keep].cpu().numpy()
        boxes = boxes[keep].cpu().numpy()
        
        if len(boxes) == 0:
            return []
        
        # Convert from cxcywh to xyxy in normalized coordinates
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        # Convert from padded coordinates to original image coordinates
        pad_w, pad_h = pad
        img_w = original_size[0] * scale
        img_h = original_size[1] * scale
        
        # Scale boxes to original image size
        x1_orig = (x1 * (img_w + pad_w) - pad_w / 2) / scale
        y1_orig = (y1 * (img_h + pad_h) - pad_h / 2) / scale
        x2_orig = (x2 * (img_w + pad_w) - pad_w / 2) / scale
        y2_orig = (y2 * (img_h + pad_h) - pad_h / 2) / scale
        
        # Clip to image boundaries
        x1_orig = np.clip(x1_orig, 0, original_size[0])
        y1_orig = np.clip(y1_orig, 0, original_size[1])
        x2_orig = np.clip(x2_orig, 0, original_size[0])
        y2_orig = np.clip(y2_orig, 0, original_size[1])
        
        # Create detections list
        detections = []
        for i in range(len(scores)):
            # Only keep valid boxes
            if x2_orig[i] > x1_orig[i] and y2_orig[i] > y1_orig[i]:
                class_id = int(labels[i])
                if class_id < len(self.config.IDX_TO_CLASS):
                    detections.append({
                        "bbox": [int(x1_orig[i]), int(y1_orig[i]), 
                                int(x2_orig[i]), int(y2_orig[i])],
                        "score": float(scores[i]),
                        "class": self.config.IDX_TO_CLASS[class_id],
                        "class_id": class_id
                    })
        
        # Apply NMS
        if len(detections) > 0:
            detections = self.nms(detections, self.config.IOU_THRESHOLD)
        
        return detections
    
    def nms(self, detections: List[Dict], iou_threshold: float) -> List[Dict]:
        """Apply Non-Maximum Suppression"""
        if len(detections) == 0:
            return detections
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x["score"], reverse=True)
        
        keep = []
        while len(detections) > 0:
            # Keep the highest confidence detection
            best = detections.pop(0)
            keep.append(best)
            
            # Remove overlapping detections
            detections = [d for d in detections if self.iou(best["bbox"], d["bbox"]) < iou_threshold]
        
        return keep
    
    @staticmethod
    def iou(box1: List[int], box2: List[int]) -> float:
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    @torch.no_grad()
    def detect(self, image: np.ndarray) -> List[Dict]:
        """Run detection on a single image"""
        original_size = (image.shape[1], image.shape[0])
        
        # Preprocess
        img_tensor, scale, pad = self.preprocess_image(image)
        img_tensor = img_tensor.to(self.config.DEVICE)
        
        # Inference
        if self.config.USE_AMP and self.config.DEVICE.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = self.model(pixel_values=img_tensor)
        else:
            outputs = self.model(pixel_values=img_tensor)
        
        # Postprocess
        detections = self.postprocess_predictions(outputs, scale, pad, original_size)
        
        return detections


# ===========================================================================
# VISUALIZATION
# ===========================================================================
def draw_detections(
    image: np.ndarray, 
    detections: List[Dict],
    config: Config
) -> np.ndarray:
    """Draw bounding boxes and labels on image"""
    img_copy = image.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_name = det["class"]
        score = det["score"]
        
        # Get color for class
        color = config.COLORS.get(class_name, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, config.LINE_THICKNESS)
        
        # Prepare label
        label = f"{class_name}: {score:.2f}"
        
        # Calculate text size
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, 1
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            img_copy, 
            (x1, y1 - text_h - baseline - 5), 
            (x1 + text_w + 5, y1), 
            color, 
            -1
        )
        
        # Draw text
        cv2.putText(
            img_copy, 
            label, 
            (x1 + 2, y1 - baseline - 3), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            config.FONT_SCALE, 
            (0, 0, 0), 
            1
        )
    
    return img_copy


def draw_fps(image: np.ndarray, fps: float) -> np.ndarray:
    """Draw FPS counter on image"""
    cv2.putText(
        image,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )
    return image


# ===========================================================================
# VIDEO PROCESSING
# ===========================================================================
def process_video(
    video_path: str,
    output_path: Optional[str],
    detector: RTDETRDetector,
    config: Config,
    display: bool = True,
    save_json: Optional[str] = None
):
    """Process video file"""
    
    # Open video capture
    if video_path.isdigit():
        cap = cv2.VideoCapture(int(video_path))
        source_name = "Webcam"
    else:
        cap = cv2.VideoCapture(str(video_path))
        source_name = Path(video_path).name
    
    if not cap.isOpened():
        raise IOError(f"Cannot open video source: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n{'='*60}")
    print(f"Processing: {source_name}")
    print(f"Resolution: {width}x{height}, FPS: {fps:.2f}")
    if total_frames > 0:
        print(f"Total frames: {total_frames}")
    print(f"{'='*60}\n")
    
    # Initialize video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Output video: {output_path}")
    
    # For JSON output
    all_detections = []
    
    # Processing loop
    frame_count = 0
    start_time = time.time()
    processing_times = []
    fps_display = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        frame_start = time.time()
        detections = detector.detect(frame)
        frame_time = time.time() - frame_start
        processing_times.append(frame_time)
        
        # Calculate real-time FPS
        if len(processing_times) > 30:
            avg_time = np.mean(processing_times[-30:])
            fps_display = 1.0 / avg_time if avg_time > 0 else 0
        
        # Draw results
        vis_frame = draw_detections(frame, detections, config)
        
        if config.SHOW_FPS:
            vis_frame = draw_fps(vis_frame, fps_display)
        
        # Save detections to JSON
        if save_json:
            frame_detections = []
            for det in detections:
                frame_detections.append({
                    "frame": frame_count,
                    "class": det["class"],
                    "confidence": det["score"],
                    "bbox": det["bbox"]
                })
            all_detections.extend(frame_detections)
        
        # Save or display
        if writer:
            writer.write(vis_frame)
        
        if display:
            cv2.imshow('RT-DETR Detection', vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
        
        # Progress report
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            if total_frames > 0:
                progress = frame_count / total_frames
                eta = elapsed / progress * (1 - progress) if progress > 0 else 0
                print(f"Frame {frame_count}/{total_frames} ({progress*100:.1f}%) - "
                      f"FPS: {fps_display:.1f} - ETA: {eta:.1f}s", end="\r")
            else:
                print(f"Frame {frame_count} - FPS: {fps_display:.1f}", end="\r")
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()
    
    # Save JSON output
    if save_json and all_detections:
        with open(save_json, 'w') as f:
            json.dump(all_detections, f, indent=2)
        print(f"\nDetections saved to: {save_json}")
    
    # Summary statistics
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_inference_time = np.mean(processing_times) * 1000 if processing_times else 0
    
    print(f"\n\n{'='*60}")
    print(f"Processing Complete!")
    print(f"Total frames: {frame_count}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Average inference time: {avg_inference_time:.1f}ms")
    print(f"{'='*60}")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description='RT-DETR Video Inference')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to trained model weights (.pth file)')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to input video file')
    parser.add_argument('--webcam', type=int, default=None,
                        help='Webcam device ID (e.g., 0)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output video file')
    parser.add_argument('--json', type=str, default=None,
                        help='Path to save detections as JSON')
    parser.add_argument('--conf-thresh', type=float, default=0.3,
                        help='Confidence threshold (default: 0.3)')
    parser.add_argument('--iou-thresh', type=float, default=0.5,
                        help='NMS IoU threshold (default: 0.5)')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable display window')
    parser.add_argument('--no-fps', action='store_true',
                        help='Hide FPS counter')
    
    args = parser.parse_args()
    
    # Update config
    config = Config()
    config.CONFIDENCE_THRESHOLD = args.conf_thresh
    config.IOU_THRESHOLD = args.iou_thresh
    config.SHOW_FPS = not args.no_fps
    
    # Check input source
    if not args.video and args.webcam is None:
        print("Error: Please specify either --video or --webcam")
        return
    
    video_source = args.video if args.video else str(args.webcam)
    
    # Initialize detector
    detector = RTDETRDetector(config)
    
    try:
        # Load model
        detector.load_model(args.weights)
        
        # Process video
        process_video(
            video_path=video_source,
            output_path=args.output,
            detector=detector,
            config=config,
            display=not args.no_display,
            save_json=args.json
        )
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
