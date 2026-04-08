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
from collections import defaultdict, Counter
import time
import psutil
import threading
import GPUtil
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple, Any

class ResourceMonitor:
    """Comprehensive resource monitoring class"""
    
    def __init__(self):
        self.monitoring = False
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        self.timestamps = []
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start resource monitoring in background"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def _monitor_loop(self):
        """Monitor system resources"""
        while self.monitoring:
            try:
                # CPU and Memory
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                
                # GPU metrics
                gpu_stats = []
                gpu_mem_stats = []
                try:
                    gpus = GPUtil.getGPUs()
                    for gpu in gpus:
                        gpu_stats.append(gpu.load * 100)
                        gpu_mem_stats.append(gpu.memoryUtil * 100)
                except:
                    gpu_stats = [0.0]
                    gpu_mem_stats = [0.0]
                
                self.cpu_usage.append(cpu_percent)
                self.memory_usage.append(memory.percent)
                self.gpu_usage.append(gpu_stats[0] if gpu_stats else 0.0)
                self.gpu_memory.append(gpu_mem_stats[0] if gpu_mem_stats else 0.0)
                self.timestamps.append(time.time())
                
                time.sleep(0.5)
            except Exception as e:
                print(f"Monitoring error: {e}")
                
    def stop_monitoring(self):
        """Stop monitoring and return statistics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            
        return self.get_statistics()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive resource statistics"""
        if not self.cpu_usage:
            return {}
            
        stats = {
            'cpu': {
                'mean': np.mean(self.cpu_usage),
                'max': np.max(self.cpu_usage),
                'min': np.min(self.cpu_usage),
                'std': np.std(self.cpu_usage),
                'median': np.median(self.cpu_usage)
            },
            'memory': {
                'mean': np.mean(self.memory_usage),
                'max': np.max(self.memory_usage),
                'min': np.min(self.memory_usage),
                'std': np.std(self.memory_usage),
                'median': np.median(self.memory_usage)
            },
            'gpu': {
                'mean': np.mean(self.gpu_usage),
                'max': np.max(self.gpu_usage),
                'min': np.min(self.gpu_usage),
                'std': np.std(self.gpu_usage),
                'median': np.median(self.gpu_usage)
            },
            'gpu_memory': {
                'mean': np.mean(self.gpu_memory),
                'max': np.max(self.gpu_memory),
                'min': np.min(self.gpu_memory),
                'std': np.std(self.gpu_memory),
                'median': np.median(self.gpu_memory)
            },
            'duration': self.timestamps[-1] - self.timestamps[0] if self.timestamps else 0,
            'sample_count': len(self.cpu_usage)
        }
        
        return stats

class COCODatasetCreator:
    def __init__(self, video_path, model_path, output_dir="prepared_dataset", fps=15, confidence_threshold=0.5):
        self.video_path = video_path
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.confidence_threshold = confidence_threshold
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        self.metrics = {
            'extraction_time': 0,
            'annotation_time': 0,
            'total_time': 0,
            'detection_stats': defaultdict(list),
            'category_counts': Counter(),
            'bbox_sizes': [],
            'confidence_scores': []
        }
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        self.images_dir = self.output_dir / "images"
        self.annotations_dir = self.output_dir / "annotations"
        self.plots_dir = self.output_dir / "analysis_plots"
        self.images_dir.mkdir(exist_ok=True)
        self.annotations_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        
        # Initialize model
        print("Loading RF-DETR Medium model...")
        model_load_start = time.time()
        self.model = RFDETRMedium(
            pretrain_weights=model_path,
            device="cpu"
        )
        self.model.optimize_for_inference()
        model_load_time = time.time() - model_load_start
        print(f"Model loaded successfully in {model_load_time:.2f} seconds!")
        
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
        
        # Category name mapping
        self.id_to_name = {cat['id']: cat['name'] for cat in self.coco_categories}

    def extract_frames(self):
        """Extract frames from video at specified FPS with resource monitoring"""
        print(f"Extracting frames from {self.video_path} at {self.fps} FPS...")
        
        # Start monitoring
        self.resource_monitor.start_monitoring()
        extraction_start = time.time()
        
        cap = cv2.VideoCapture(self.video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(original_fps / self.fps) if original_fps > 0 else 1
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / original_fps if original_fps > 0 else 0
        
        frames = []
        frame_count = 0
        saved_count = 0
        
        # Progress bar
        pbar = tqdm(total=total_frames, desc="Extracting frames", unit="frames")
        
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
                    'original_index': frame_count,
                    'width': frame.shape[1],
                    'height': frame.shape[0]
                })
                saved_count += 1
            
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        # Stop monitoring
        extraction_stats = self.resource_monitor.stop_monitoring()
        self.metrics['extraction_time'] = time.time() - extraction_start
        
        print(f"\nExtraction Statistics:")
        print(f"   • Total frames extracted: {len(frames)}")
        print(f"   • Original video FPS: {original_fps:.2f}")
        print(f"   • Video duration: {video_duration:.2f} seconds")
        print(f"   • Extraction time: {self.metrics['extraction_time']:.2f} seconds")
        print(f"   • Average extraction speed: {len(frames)/self.metrics['extraction_time']:.2f} FPS")
        print(f"   • CPU usage during extraction: {extraction_stats.get('cpu', {}).get('mean', 0):.1f}%")
        print(f"   • Memory usage during extraction: {extraction_stats.get('memory', {}).get('mean', 0):.1f}%")
        
        return frames

    def shuffle_frames(self, frames):
        """Shuffle frames using conventional hardcoded algorithm with ID changes"""
        print("Shuffling frames with ID reassignment...")
        
        # Conventional Fisher-Yates shuffle implementation
        n = len(frames)
        shuffled_indices = list(range(n))
        
        # Hardcoded shuffle algorithm
        for i in range(n-1, 0, -1):
            # Use a deterministic but pseudo-random approach
            j = (i * 7 + 13) % (i + 1)  # Simple deterministic pseudo-random
            shuffled_indices[i], shuffled_indices[j] = shuffled_indices[j], shuffled_indices[i]
        
        # Create new shuffled list with updated IDs and filenames
        shuffled_frames = []
        for new_pos, original_idx in enumerate(shuffled_indices):
            original_frame = frames[original_idx]
            
            # Create new filename with shuffled ID
            new_filename = f"frame_{new_pos:06d}.jpg"
            new_path = self.images_dir / new_filename
            
            # Rename the actual file
            if original_frame['path'].exists():
                original_frame['path'].rename(new_path)
            
            # Update frame info with new ID and filename
            shuffled_frame = {
                'filename': new_filename,
                'path': new_path,
                'original_index': original_frame['original_index'],
                'width': original_frame['width'],
                'height': original_frame['height'],
                'shuffled_id': new_pos + 1,  # New 1-based ID
                'original_id': original_idx + 1  # Original 1-based ID
            }
            shuffled_frames.append(shuffled_frame)
        
        print(f"Frames shuffled successfully! Renamed {len(frames)} files with new IDs.")
        return shuffled_frames

    def annotate_frames(self, frames):
        """Annotate frames using RF-DETR with comprehensive monitoring"""
        print("Annotating frames with RF-DETR...")
        
        # Start monitoring
        self.resource_monitor.start_monitoring()
        annotation_start = time.time()
        
        annotation_id = 1
        detected_categories = set()
        inference_times = []
        
        # Progress bar
        pbar = tqdm(total=len(frames), desc="Annotating frames", unit="frames")
        
        for frame_info in frames:
            frame_start = time.time()
            
            # Load and convert frame
            frame = cv2.imread(str(frame_info['path']))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Run inference
            detections = self.model.predict(pil_image, threshold=self.confidence_threshold)
            inference_time = time.time() - frame_start
            inference_times.append(inference_time)
            
            # Add image info to COCO data using shuffled_id
            height, width = frame.shape[:2]
            image_info = {
                "id": frame_info['shuffled_id'],  # Use shuffled ID
                "width": width,
                "height": height,
                "file_name": frame_info['filename'],
                "license": 1,
                "date_captured": datetime.now().isoformat(),
                "original_frame_index": frame_info['original_index'],  # Track original position
                "original_id": frame_info['original_id']
            }
            self.coco_data["images"].append(image_info)
            
            # Process detections
            frame_detections = len(detections)
            self.metrics['detection_stats']['detections_per_frame'].append(frame_detections)
            
            # Add annotations
            for j in range(len(detections)):
                if detections.confidence[j] >= self.confidence_threshold:
                    bbox = detections.xyxy[j]  # [x1, y1, x2, y2]
                    category_id = detections.class_id[j] + 1  # Convert to 1-based indexing
                    confidence = detections.confidence[j]
                    
                    # Convert to COCO bbox format [x, y, width, height]
                    x, y, x2, y2 = bbox
                    width_bbox = x2 - x
                    height_bbox = y2 - y
                    area = width_bbox * height_bbox
                    
                    # Collect metrics
                    self.metrics['category_counts'][category_id] += 1
                    self.metrics['bbox_sizes'].append(area)
                    self.metrics['confidence_scores'].append(confidence)
                    
                    annotation = {
                        "id": annotation_id,
                        "image_id": frame_info['shuffled_id'],  # Use shuffled ID
                        "category_id": int(category_id),
                        "bbox": [float(x), float(y), float(width_bbox), float(height_bbox)],
                        "area": float(area),
                        "iscrowd": 0,
                        "confidence": float(confidence)
                    }
                    self.coco_data["annotations"].append(annotation)
                    annotation_id += 1
                    detected_categories.add(int(category_id))
            
            pbar.update(1)
        
        pbar.close()
        
        # Stop monitoring
        annotation_stats = self.resource_monitor.stop_monitoring()
        self.metrics['annotation_time'] = time.time() - annotation_start
        
        print(f"\nAnnotation Statistics:")
        print(f"   • Total annotations: {len(self.coco_data['annotations'])}")
        print(f"   • Unique categories detected: {len(detected_categories)}")
        print(f"   • Annotation time: {self.metrics['annotation_time']:.2f} seconds")
        print(f"   • Average annotation speed: {len(frames)/self.metrics['annotation_time']:.2f} FPS")
        print(f"   • Average inference time per frame: {np.mean(inference_times):.3f} seconds")
        print(f"   • CPU usage during annotation: {annotation_stats.get('cpu', {}).get('mean', 0):.1f}%")
        print(f"   • Memory usage during annotation: {annotation_stats.get('memory', {}).get('mean', 0):.1f}%")
        print(f"   • Average detections per frame: {np.mean(self.metrics['detection_stats']['detections_per_frame']):.1f}")
        
        return detected_categories

    def create_train_val_test_split(self, frames, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """Create train/validation/test splits using shuffled IDs"""
        print(f"Creating train ({train_ratio*100}%)/val ({val_ratio*100}%)/test ({test_ratio*100}%) splits...")
        
        total_frames = len(frames)
        train_end = int(total_frames * train_ratio)
        val_end = train_end + int(total_frames * val_ratio)
        
        # Use shuffled IDs from frame data
        train_indices = [frame['shuffled_id'] for frame in frames[:train_end]]
        val_indices = [frame['shuffled_id'] for frame in frames[train_end:val_end]]
        test_indices = [frame['shuffled_id'] for frame in frames[val_end:]]
        
        splits = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
        
        print(f"   • Train: {len(train_indices)} frames")
        print(f"   • Validation: {len(val_indices)} frames")
        print(f"   • Test: {len(test_indices)} frames")
        
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
            
            print(f"   • {split_name}: {len(split_data['images'])} images, {len(split_data['annotations'])} annotations")

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
        
        print(f"   • Dataset configuration saved to: {yaml_path}")

    def generate_eda_plots(self):
        """Generate comprehensive EDA plots"""
        print("Generating Exploratory Data Analysis plots...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Category Distribution
        if self.metrics['category_counts']:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Top 20 categories
            top_categories = self.metrics['category_counts'].most_common(20)
            categories = [self.id_to_name.get(cat_id, f"Class_{cat_id}") for cat_id, _ in top_categories]
            counts = [count for _, count in top_categories]
            
            ax1.barh(categories, counts)
            ax1.set_title('Top 20 Detected Categories', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Count')
            ax1.set_ylabel('Category')
            
            # Category distribution pie chart
            other_count = sum(count for _, count in self.metrics['category_counts'].items() if count < 50)
            top_10 = self.metrics['category_counts'].most_common(10)
            pie_labels = [self.id_to_name.get(cat_id, f"Class_{cat_id}") for cat_id, _ in top_10]
            pie_counts = [count for _, count in top_10]
            
            if other_count > 0:
                pie_labels.append("Others")
                pie_counts.append(other_count)
            
            ax2.pie(pie_counts, labels=pie_labels, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Category Distribution (Top 10)', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'category_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Bounding Box Analysis
        if self.metrics['bbox_sizes']:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Bbox size distribution
            ax1.hist(self.metrics['bbox_sizes'], bins=50, alpha=0.7, edgecolor='black')
            ax1.set_title('Bounding Box Area Distribution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Area (pixels²)')
            ax1.set_ylabel('Frequency')
            ax1.set_xscale('log')
            
            # Confidence score distribution
            ax2.hist(self.metrics['confidence_scores'], bins=50, alpha=0.7, edgecolor='black', color='green')
            ax2.set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Confidence Score')
            ax2.set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'bbox_confidence_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Detections per Frame
        if self.metrics['detection_stats']['detections_per_frame']:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            detections_per_frame = self.metrics['detection_stats']['detections_per_frame']
            
            # Time series of detections
            ax1.plot(detections_per_frame, alpha=0.7)
            ax1.set_title('Detections per Frame (Time Series)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Frame Index')
            ax1.set_ylabel('Number of Detections')
            
            # Distribution of detections per frame
            ax2.hist(detections_per_frame, bins=30, alpha=0.7, edgecolor='black', color='orange')
            ax2.set_title('Detections per Frame Distribution', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Number of Detections')
            ax2.set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'detections_per_frame.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("   EDA plots saved to analysis_plots/ directory")

    def generate_comprehensive_report(self):
        """Generate detailed EDA report"""
        print("Generating comprehensive EDA report...")
        
        report = {
            "dataset_overview": {
                "total_images": len(self.coco_data['images']),
                "total_annotations": len(self.coco_data['annotations']),
                "unique_categories": len(self.metrics['category_counts']),
                "avg_annotations_per_image": len(self.coco_data['annotations']) / len(self.coco_data['images']) if self.coco_data['images'] else 0,
                "extraction_fps": self.fps,
                "confidence_threshold": self.confidence_threshold
            },
            "performance_metrics": {
                "extraction_time_seconds": self.metrics['extraction_time'],
                "annotation_time_seconds": self.metrics['annotation_time'],
                "total_processing_time_seconds": self.metrics['extraction_time'] + self.metrics['annotation_time'],
                "extraction_speed_fps": len(self.coco_data['images']) / self.metrics['extraction_time'] if self.metrics['extraction_time'] > 0 else 0,
                "annotation_speed_fps": len(self.coco_data['images']) / self.metrics['annotation_time'] if self.metrics['annotation_time'] > 0 else 0
            },
            "detection_statistics": {
                "avg_detections_per_frame": np.mean(self.metrics['detection_stats']['detections_per_frame']) if self.metrics['detection_stats']['detections_per_frame'] else 0,
                "max_detections_per_frame": np.max(self.metrics['detection_stats']['detections_per_frame']) if self.metrics['detection_stats']['detections_per_frame'] else 0,
                "min_detections_per_frame": np.min(self.metrics['detection_stats']['detections_per_frame']) if self.metrics['detection_stats']['detections_per_frame'] else 0,
                "std_detections_per_frame": np.std(self.metrics['detection_stats']['detections_per_frame']) if self.metrics['detection_stats']['detections_per_frame'] else 0,
                "avg_confidence_score": np.mean(self.metrics['confidence_scores']) if self.metrics['confidence_scores'] else 0,
                "avg_bbox_area": np.mean(self.metrics['bbox_sizes']) if self.metrics['bbox_sizes'] else 0,
                "median_bbox_area": np.median(self.metrics['bbox_sizes']) if self.metrics['bbox_sizes'] else 0
            },
            "category_analysis": {
                "top_10_categories": [
                    {
                        "category_id": cat_id,
                        "category_name": self.id_to_name.get(cat_id, f"Class_{cat_id}"),
                        "count": count,
                        "percentage": (count / len(self.coco_data['annotations'])) * 100
                    }
                    for cat_id, count in self.metrics['category_counts'].most_common(10)
                ],
                "category_distribution": {
                    self.id_to_name.get(cat_id, f"Class_{cat_id}"): count 
                    for cat_id, count in self.metrics['category_counts'].items()
                }
            },
            "resource_utilization": {
                "system_info": {
                    "cpu_count": psutil.cpu_count(),
                    "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                    "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}"
                },
                "processing_efficiency": {
                    "frames_per_second_overall": len(self.coco_data['images']) / (self.metrics['extraction_time'] + self.metrics['annotation_time']),
                    "memory_efficiency_mb_per_frame": (psutil.virtual_memory().used / (1024**2)) / len(self.coco_data['images']) if self.coco_data['images'] else 0
                }
            }
        }
        
        # Save report as JSON
        with open(self.output_dir / 'eda_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create human-readable summary
        summary = f"""
# 📊 Dataset Creation EDA Report

## 🎯 Dataset Overview
- **Total Images**: {report['dataset_overview']['total_images']:,}
- **Total Annotations**: {report['dataset_overview']['total_annotations']:,}
- **Unique Categories**: {report['dataset_overview']['unique_categories']}
- **Avg Annotations per Image**: {report['dataset_overview']['avg_annotations_per_image']:.2f}
- **Extraction FPS**: {report['dataset_overview']['extraction_fps']}
- **Confidence Threshold**: {report['dataset_overview']['confidence_threshold']}

## ⚡ Performance Metrics
- **Extraction Time**: {report['performance_metrics']['extraction_time_seconds']:.2f} seconds
- **Annotation Time**: {report['performance_metrics']['annotation_time_seconds']:.2f} seconds
- **Total Processing Time**: {report['performance_metrics']['total_processing_time_seconds']:.2f} seconds
- **Extraction Speed**: {report['performance_metrics']['extraction_speed_fps']:.2f} FPS
- **Annotation Speed**: {report['performance_metrics']['annotation_speed_fps']:.2f} FPS

## 🎯 Detection Statistics
- **Avg Detections per Frame**: {report['detection_statistics']['avg_detections_per_frame']:.2f}
- **Max Detections per Frame**: {report['detection_statistics']['max_detections_per_frame']}
- **Min Detections per Frame**: {report['detection_statistics']['min_detections_per_frame']}
- **Std Detections per Frame**: {report['detection_statistics']['std_detections_per_frame']:.2f}
- **Avg Confidence Score**: {report['detection_statistics']['avg_confidence_score']:.3f}
- **Avg Bbox Area**: {report['detection_statistics']['avg_bbox_area']:.0f} pixels²
- **Median Bbox Area**: {report['detection_statistics']['median_bbox_area']:.0f} pixels²

## 🏆 Top 10 Categories
"""
        
        for i, cat in enumerate(report['category_analysis']['top_10_categories'], 1):
            summary += f"{i}. **{cat['category_name']}**: {cat['count']:,} annotations ({cat['percentage']:.1f}%)\n"
        
        summary += f"""
## 💻 System Information
- **CPU Cores**: {report['resource_utilization']['system_info']['cpu_count']}
- **Total Memory**: {report['resource_utilization']['system_info']['memory_total_gb']:.1f} GB
- **Python Version**: {report['resource_utilization']['system_info']['python_version']}

## 📈 Processing Efficiency
- **Overall FPS**: {report['resource_utilization']['processing_efficiency']['frames_per_second_overall']:.2f}
- **Memory per Frame**: {report['resource_utilization']['processing_efficiency']['memory_efficiency_mb_per_frame']:.2f} MB

---
*Report generated on: {datetime.now().isoformat()}*
"""
        
        with open(self.output_dir / 'eda_summary.md', 'w') as f:
            f.write(summary)
        
        print("   EDA report saved as JSON and Markdown")

    def create_dataset(self):
        """Main pipeline to create the dataset with comprehensive analysis"""
        print("Starting comprehensive dataset creation pipeline...")
        total_start = time.time()
        
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
        
        # Step 7: Generate EDA plots
        self.generate_eda_plots()
        
        # Step 8: Generate comprehensive report
        self.generate_comprehensive_report()
        
        self.metrics['total_time'] = time.time() - total_start
        
        print(f"\nDataset creation completed successfully!")
        print(f"Output directory: {self.output_dir}")
        print(f"Total processing time: {self.metrics['total_time']:.2f} seconds")
        print(f"Total images: {len(self.coco_data['images']):,}")
        print(f"Total annotations: {len(self.coco_data['annotations']):,}")
        print(f"Categories detected: {len(detected_categories)}")
        print(f"EDA report and plots generated!")
        
        return self.output_dir

if __name__ == "__main__":
    # Configuration
    video_path = "test.mp4"
    model_path = "/home/pragesh-shrestha/Desktop/binayak_sir/rf-detr-medium.pth"
    output_dir = "prepared_dataset"
    fps = 15
    confidence_threshold = 0.5
    
    # Create dataset with comprehensive analysis
    creator = COCODatasetCreator(
        video_path=video_path,
        model_path=model_path,
        output_dir=output_dir,
        fps=fps,
        confidence_threshold=confidence_threshold
    )
    
    dataset_path = creator.create_dataset()
    print(f"\nDataset ready at: {dataset_path}")
    print(f"Check the 'analysis_plots' directory for visualizations!")
    print(f"Check 'eda_summary.md' for detailed insights!")
