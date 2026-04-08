import os
import json
import random
import yaml
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from rfdetr import RFDETRMedium
import supervision as sv
from datetime import datetime
from collections import defaultdict

class COCODatasetCreator:
    def __init__(self, video_path, model_path, output_dir="prepared_dataset", fps=15, confidence_threshold=0.5):
        self.video_path = video_path
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.confidence_threshold = confidence_threshold
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        self.images_dir = self.output_dir / "images"
        self.annotations_dir = self.output_dir / "annotations"
        self.images_dir.mkdir(exist_ok=True)
        self.annotations_dir.mkdir(exist_ok=True)
        
        # Initialize model
        print("Loading RF-DETR Medium model...")
        self.model = RFDETRMedium(
            pretrain_weights=model_path,
            device="cpu"
        )
        self.model.optimize_for_inference()
        print("Model loaded successfully!")
        
        # COCO dataset structure
        self.coco_data = {
            "info": {
                "description": "Dataset created from video using RF-DETR",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "RF-DETR Auto-Annotation",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Category mapping (RF-DETR uses COCO classes)
        self.coco_categories = [
            {"id": 1, "name": "person", "supercategory": "person"},
            {"id": 2, "name": "bicycle", "supercategory": "vehicle"},
            {"id": 3, "name": "car", "supercategory": "vehicle"},
            {"id": 4, "name": "motorcycle", "supercategory": "vehicle"},
            {"id": 5, "name": "airplane", "supercategory": "vehicle"},
            {"id": 6, "name": "bus", "supercategory": "vehicle"},
            {"id": 7, "name": "train", "supercategory": "vehicle"},
            {"id": 8, "name": "truck", "supercategory": "vehicle"},
            {"id": 9, "name": "boat", "supercategory": "vehicle"},
            {"id": 10, "name": "traffic light", "supercategory": "outdoor"},
            {"id": 11, "name": "fire hydrant", "supercategory": "outdoor"},
            {"id": 13, "name": "stop sign", "supercategory": "outdoor"},
            {"id": 14, "name": "parking meter", "supercategory": "outdoor"},
            {"id": 15, "name": "bench", "supercategory": "outdoor"},
            {"id": 16, "name": "bird", "supercategory": "animal"},
            {"id": 17, "name": "cat", "supercategory": "animal"},
            {"id": 18, "name": "dog", "supercategory": "animal"},
            {"id": 19, "name": "horse", "supercategory": "animal"},
            {"id": 20, "name": "sheep", "supercategory": "animal"},
            {"id": 21, "name": "cow", "supercategory": "animal"},
            {"id": 22, "name": "elephant", "supercategory": "animal"},
            {"id": 23, "name": "bear", "supercategory": "animal"},
            {"id": 24, "name": "zebra", "supercategory": "animal"},
            {"id": 25, "name": "giraffe", "supercategory": "animal"},
            {"id": 27, "name": "backpack", "supercategory": "accessory"},
            {"id": 28, "name": "umbrella", "supercategory": "accessory"},
            {"id": 31, "name": "handbag", "supercategory": "accessory"},
            {"id": 32, "name": "tie", "supercategory": "accessory"},
            {"id": 33, "name": "suitcase", "supercategory": "accessory"},
            {"id": 34, "name": "frisbee", "supercategory": "sports"},
            {"id": 35, "name": "skis", "supercategory": "sports"},
            {"id": 36, "name": "snowboard", "supercategory": "sports"},
            {"id": 37, "name": "sports ball", "supercategory": "sports"},
            {"id": 38, "name": "kite", "supercategory": "sports"},
            {"id": 39, "name": "baseball bat", "supercategory": "sports"},
            {"id": 40, "name": "baseball glove", "supercategory": "sports"},
            {"id": 41, "name": "skateboard", "supercategory": "sports"},
            {"id": 42, "name": "surfboard", "supercategory": "sports"},
            {"id": 43, "name": "tennis racket", "supercategory": "sports"},
            {"id": 44, "name": "bottle", "supercategory": "kitchen"},
            {"id": 46, "name": "wine glass", "supercategory": "kitchen"},
            {"id": 47, "name": "cup", "supercategory": "kitchen"},
            {"id": 48, "name": "fork", "supercategory": "kitchen"},
            {"id": 49, "name": "knife", "supercategory": "kitchen"},
            {"id": 50, "name": "spoon", "supercategory": "kitchen"},
            {"id": 51, "name": "bowl", "supercategory": "kitchen"},
            {"id": 52, "name": "banana", "supercategory": "food"},
            {"id": 53, "name": "apple", "supercategory": "food"},
            {"id": 54, "name": "sandwich", "supercategory": "food"},
            {"id": 55, "name": "orange", "supercategory": "food"},
            {"id": 56, "name": "broccoli", "supercategory": "food"},
            {"id": 57, "name": "carrot", "supercategory": "food"},
            {"id": 58, "name": "hot dog", "supercategory": "food"},
            {"id": 59, "name": "pizza", "supercategory": "food"},
            {"id": 60, "name": "donut", "supercategory": "food"},
            {"id": 61, "name": "cake", "supercategory": "food"},
            {"id": 62, "name": "chair", "supercategory": "furniture"},
            {"id": 63, "name": "couch", "supercategory": "furniture"},
            {"id": 64, "name": "potted plant", "supercategory": "furniture"},
            {"id": 65, "name": "bed", "supercategory": "furniture"},
            {"id": 67, "name": "dining table", "supercategory": "furniture"},
            {"id": 70, "name": "toilet", "supercategory": "furniture"},
            {"id": 72, "name": "tv", "supercategory": "electronic"},
            {"id": 73, "name": "laptop", "supercategory": "electronic"},
            {"id": 74, "name": "mouse", "supercategory": "electronic"},
            {"id": 75, "name": "remote", "supercategory": "electronic"},
            {"id": 76, "name": "keyboard", "supercategory": "electronic"},
            {"id": 77, "name": "cell phone", "supercategory": "electronic"},
            {"id": 78, "name": "microwave", "supercategory": "electronic"},
            {"id": 79, "name": "oven", "supercategory": "electronic"},
            {"id": 80, "name": "toaster", "supercategory": "electronic"},
            {"id": 81, "name": "sink", "supercategory": "electronic"},
            {"id": 82, "name": "refrigerator", "supercategory": "electronic"},
            {"id": 84, "name": "book", "supercategory": "furniture"},
            {"id": 85, "name": "clock", "supercategory": "furniture"},
            {"id": 86, "name": "vase", "supercategory": "furniture"},
            {"id": 87, "name": "scissors", "supercategory": "furniture"},
            {"id": 88, "name": "teddy bear", "supercategory": "furniture"},
            {"id": 89, "name": "hair drier", "supercategory": "furniture"},
            {"id": 90, "name": "toothbrush", "supercategory": "furniture"}
        ]
        self.coco_data["categories"] = self.coco_categories

    def extract_frames(self):
        """Extract frames from video at specified FPS"""
        print(f"Extracting frames from {self.video_path} at {self.fps} FPS...")
        
        cap = cv2.VideoCapture(self.video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(original_fps / self.fps) if original_fps > 0 else 1
        
        frames = []
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frame_filename = f"frame_{saved_count:06d}.jpg"
                frame_path = self.images_dir / frame_filename
                
                # Save frame
                cv2.imwrite(str(frame_path), frame)
                frames.append({
                    'filename': frame_filename,
                    'path': frame_path,
                    'original_index': frame_count
                })
                saved_count += 1
                
                if saved_count % 100 == 0:
                    print(f"Extracted {saved_count} frames...")
            
            frame_count += 1
        
        cap.release()
        print(f"Total frames extracted: {len(frames)}")
        return frames

    def shuffle_frames(self, frames):
        """Shuffle frames randomly"""
        print("Shuffling frames...")
        random.shuffle(frames)
        print("Frames shuffled successfully!")
        return frames

    def annotate_frames(self, frames):
        """Annotate frames using RF-DETR"""
        print("Annotating frames with RF-DETR...")
        
        annotation_id = 1
        detected_categories = set()
        
        for i, frame_info in enumerate(frames):
            if i % 50 == 0:
                print(f"Annotated {i}/{len(frames)} frames...")
            
            # Load and convert frame
            frame = cv2.imread(str(frame_info['path']))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Run inference
            detections = self.model.predict(pil_image, threshold=self.confidence_threshold)
            
            # Add image info to COCO data
            height, width = frame.shape[:2]
            image_info = {
                "id": i + 1,
                "width": width,
                "height": height,
                "file_name": frame_info['filename'],
                "license": 1,
                "date_captured": datetime.now().isoformat()
            }
            self.coco_data["images"].append(image_info)
            
            # Add annotations
            for j in range(len(detections)):
                if detections.confidence[j] >= self.confidence_threshold:
                    bbox = detections.xyxy[j]  # [x1, y1, x2, y2]
                    category_id = detections.class_id[j] + 1  # Convert to 1-based indexing
                    
                    # Convert to COCO bbox format [x, y, width, height]
                    x, y, x2, y2 = bbox
                    width_bbox = x2 - x
                    height_bbox = y2 - y
                    area = width_bbox * height_bbox
                    
                    annotation = {
                        "id": annotation_id,
                        "image_id": i + 1,
                        "category_id": int(category_id),
                        "bbox": [float(x), float(y), float(width_bbox), float(height_bbox)],
                        "area": float(area),
                        "iscrowd": 0
                    }
                    self.coco_data["annotations"].append(annotation)
                    annotation_id += 1
                    detected_categories.add(int(category_id))
        
        print(f"Annotation completed! Total annotations: {len(self.coco_data['annotations'])}")
        print(f"Detected categories: {len(detected_categories)}")
        return detected_categories

    def create_train_val_test_split(self, frames, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """Create train/validation/test splits"""
        print(f"Creating train ({train_ratio*100}%)/val ({val_ratio*100}%)/test ({test_ratio*100}%) splits...")
        
        total_frames = len(frames)
        train_end = int(total_frames * train_ratio)
        val_end = train_end + int(total_frames * val_ratio)
        
        train_indices = list(range(1, train_end + 1))
        val_indices = list(range(train_end + 1, val_end + 1))
        test_indices = list(range(val_end + 1, total_frames + 1))
        
        splits = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
        
        print(f"Train: {len(train_indices)} frames")
        print(f"Validation: {len(val_indices)} frames")
        print(f"Test: {len(test_indices)} frames")
        
        return splits

    def save_coco_annotations(self, splits):
        """Save COCO annotations for each split"""
        print("Saving COCO annotations...")
        
        for split_name, image_ids in splits.items():
            split_data = {
                "info": self.coco_data["info"],
                "licenses": self.coco_data["licenses"],
                "categories": self.coco_data["categories"],
                "images": [img for img in self.coco_data["images"] if img["id"] in image_ids],
                "annotations": [ann for ann in self.coco_data["annotations"] if ann["image_id"] in image_ids]
            }
            
            annotation_path = self.annotations_dir / f"{split_name}.json"
            with open(annotation_path, 'w') as f:
                json.dump(split_data, f, indent=2)
            
            print(f"Saved {split_name} annotations: {len(split_data['images'])} images, {len(split_data['annotations'])} annotations")

    def create_dataset_yaml(self, splits):
        """Create dataset configuration YAML"""
        print("Creating dataset configuration YAML...")
        
        # Get unique categories from annotations
        category_names = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        
        yaml_config = {
            'dataset_info': {
                'name': 'Video-derived COCO Dataset',
                'description': 'Dataset created from video using RF-DETR auto-annotation',
                'total_images': len(self.coco_data['images']),
                'total_annotations': len(self.coco_data['annotations']),
                'categories': len(self.coco_data['categories']),
                'created_date': datetime.now().isoformat()
            },
            'paths': {
                'root': str(self.output_dir),
                'images': str(self.images_dir),
                'annotations': str(self.annotations_dir)
            },
            'splits': {
                'train': {
                    'images': len([img for img in self.coco_data['images'] if img['id'] in splits['train']]),
                    'annotations': len([ann for ann in self.coco_data['annotations'] if ann['image_id'] in splits['train']]),
                    'annotation_file': str(self.annotations_dir / 'train.json')
                },
                'val': {
                    'images': len([img for img in self.coco_data['images'] if img['id'] in splits['val']]),
                    'annotations': len([ann for ann in self.coco_data['annotations'] if ann['image_id'] in splits['val']]),
                    'annotation_file': str(self.annotations_dir / 'val.json')
                },
                'test': {
                    'images': len([img for img in self.coco_data['images'] if img['id'] in splits['test']]),
                    'annotations': len([ann for ann in self.coco_data['annotations'] if ann['image_id'] in splits['test']]),
                    'annotation_file': str(self.annotations_dir / 'test.json')
                }
            },
            'categories': category_names,
            'training_config': {
                'input_format': 'coco',
                'target_format': 'coco',
                'confidence_threshold': self.confidence_threshold,
                'extraction_fps': self.fps
            }
        }
        
        yaml_path = self.output_dir / 'dataset_config.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False, indent=2)
        
        print(f"Dataset configuration saved to: {yaml_path}")

    def create_dataset(self):
        """Main pipeline to create the dataset"""
        print("Starting dataset creation pipeline...")
        
        # Step 1: Extract frames
        frames = self.extract_frames()
        
        # Step 2: Shuffle frames
        shuffled_frames = self.shuffle_frames(frames)
        
        # Step 3: Annotate frames
        detected_categories = self.annotate_frames(shuffled_frames)
        
        # Step 4: Create train/val/test splits
        splits = self.create_train_val_test_split(shuffled_frames)
        
        # Step 5: Save COCO annotations
        self.save_coco_annotations(splits)
        
        # Step 6: Create dataset YAML
        self.create_dataset_yaml(splits)
        
        print(f"\nDataset creation completed!")
        print(f"Output directory: {self.output_dir}")
        print(f"Total images: {len(self.coco_data['images'])}")
        print(f"Total annotations: {len(self.coco_data['annotations'])}")
        print(f"Categories detected: {len(detected_categories)}")
        
        return self.output_dir

if __name__ == "__main__":
    # Configuration
    video_path = "test.mp4"
    model_path = "/home/pragesh-shrestha/Desktop/binayak_sir/rf-detr-medium.pth"
    output_dir = "prepared_dataset"
    fps = 15
    confidence_threshold = 0.5
    
    # Create dataset
    creator = COCODatasetCreator(
        video_path=video_path,
        model_path=model_path,
        output_dir=output_dir,
        fps=fps,
        confidence_threshold=confidence_threshold
    )
    
    dataset_path = creator.create_dataset()
    print(f"\nDataset ready at: {dataset_path}")
