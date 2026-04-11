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

class YOLODatasetCreator:
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
        self.labels_dir = self.output_dir / "labels"
        self.plots_dir = self.output_dir / "analysis_plots"
        self.images_dir.mkdir(exist_ok=True)
        self.labels_dir.mkdir(exist_ok=True)
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
        
        # YOLO dataset structure
        self.yolo_data = {
            "info": {
                "description": "Dataset created from video using RF-DETR",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "RF-DETR Auto-Annotation",
                "date_created": datetime.now().isoformat()
            },
            "images": [],
            "annotations": []
        }
        
        # YOLO class names (RF-DETR uses COCO classes)
        self.yolo_classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        ]
        
        # Category mappings
        self.id_to_name = {i: name for i, name in enumerate(self.yolo_classes)}
        self.name_to_id = {name: i for i, name in enumerate(self.yolo_classes)}

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
        """Shuffle frames randomly"""
        print("Shuffling frames...")
        random.shuffle(frames)
        print("Frames shuffled successfully!")
        return frames

    def annotate_frames(self, frames):
        """Annotate frames using RF-DETR and create YOLO format annotations"""
        print("Annotating frames with RF-DETR...")
        
        # Start monitoring
        self.resource_monitor.start_monitoring()
        annotation_start = time.time()
        
        detected_categories = set()
        inference_times = []
        total_annotations = 0
        
        # Progress bar
        pbar = tqdm(total=len(frames), desc="Annotating frames", unit="frames")
        
        for i, frame_info in enumerate(frames):
            frame_start = time.time()
            
            # Load and convert frame
            frame = cv2.imread(str(frame_info['path']))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Run inference
            detections = self.model.predict(pil_image, threshold=self.confidence_threshold)
            inference_time = time.time() - frame_start
            inference_times.append(inference_time)
            
            # Get image dimensions for YOLO format
            height, width = frame.shape[:2]
            
            # Add image info to YOLO data
            image_info = {
                "id": i + 1,
                "width": width,
                "height": height,
                "file_name": frame_info['filename'],
                "path": str(frame_info['path'])
            }
            self.yolo_data["images"].append(image_info)
            
            # Create YOLO annotation file
            label_filename = frame_info['filename'].replace('.jpg', '.txt')
            label_path = self.labels_dir / label_filename
            
            # Process detections and create YOLO format annotations
            frame_annotations = []
            frame_detections = len(detections)
            self.metrics['detection_stats']['detections_per_frame'].append(frame_detections)
            
            for j in range(len(detections)):
                if detections.confidence[j] >= self.confidence_threshold:
                    bbox = detections.xyxy[j]  # [x1, y1, x2, y2]
                    category_id = int(detections.class_id[j]) - 2  # RF-DETR class ID
                    print(category_id)
                    confidence = detections.confidence[j]
                    
                    # Convert to YOLO format [class_id, x_center, y_center, width, height] (normalized)
                    x1, y1, x2, y2 = bbox
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    
                    # Normalize coordinates
                    x_center = (x1 + x2) / 2.0 / width
                    y_center = (y1 + y2) / 2.0 / height
                    norm_width = bbox_width / width
                    norm_height = bbox_height / height
                    
                    # YOLO format: class_id x_center y_center width height
                    yolo_annotation = f"{category_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
                    frame_annotations.append(yolo_annotation)
                    
                    # Collect metrics
                    area = bbox_width * bbox_height
                    self.metrics['category_counts'][category_id] += 1
                    self.metrics['bbox_sizes'].append(area)
                    self.metrics['confidence_scores'].append(confidence)
                    
                    # Store annotation data for analysis
                    annotation_data = {
                        "image_id": i + 1,
                        "category_id": category_id,
                        "bbox": [float(x1), float(y1), float(bbox_width), float(bbox_height)],
                        "area": float(area),
                        "confidence": float(confidence),
                        "yolo_format": [category_id, x_center, y_center, norm_width, norm_height]
                    }
                    self.yolo_data["annotations"].append(annotation_data)
                    total_annotations += 1
                    detected_categories.add(category_id)
            
            # Write YOLO annotation file
            with open(label_path, 'w') as f:
                f.write('\n'.join(frame_annotations))
            
            pbar.update(1)
        
        pbar.close()
        
        # Stop monitoring
        annotation_stats = self.resource_monitor.stop_monitoring()
        self.metrics['annotation_time'] = time.time() - annotation_start
        
        print(f"\nAnnotation Statistics:")
        print(f"   Total annotations: {total_annotations}")
        print(f"   Unique categories detected: {len(detected_categories)}")
        print(f"   Annotation time: {self.metrics['annotation_time']:.2f} seconds")
        print(f"   Average annotation speed: {len(frames)/self.metrics['annotation_time']:.2f} FPS")
        print(f"   Average inference time per frame: {np.mean(inference_times):.3f} seconds")
        print(f"   CPU usage during annotation: {annotation_stats.get('cpu', {}).get('mean', 0):.1f}%")
        print(f"   Memory usage during annotation: {annotation_stats.get('memory', {}).get('mean', 0):.1f}%")
        print(f"   Average detections per frame: {np.mean(self.metrics['detection_stats']['detections_per_frame']):.1f}")
        
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
        
        print(f"   • Train: {len(train_indices)} frames")
        print(f"   • Validation: {len(val_indices)} frames")
        print(f"   • Test: {len(test_indices)} frames")
        
        return splits

    def save_yolo_annotations(self, splits):
        """Organize YOLO annotations and images for train/val/test splits"""
        print("Organizing YOLO dataset for train/val/test splits...")
        
        # Create subdirectories for each split
        for split_name in ['train', 'val', 'test']:
            split_images_dir = self.output_dir / split_name / "images"
            split_labels_dir = self.output_dir / split_name / "labels"
            split_images_dir.mkdir(parents=True, exist_ok=True)
            split_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Move files to appropriate split directories
        for split_name, image_ids in splits.items():
            split_images_dir = self.output_dir / split_name / "images"
            split_labels_dir = self.output_dir / split_name / "labels"
            
            # Get images for this split
            split_images = [img for img in self.yolo_data["images"] if img["id"] in image_ids]
            
            moved_count = 0
            for img_info in split_images:
                # Source paths
                src_image_path = Path(img_info["path"])
                src_label_path = self.labels_dir / img_info["file_name"].replace('.jpg', '.txt')
                
                # Destination paths
                dst_image_path = split_images_dir / img_info["file_name"]
                dst_label_path = split_labels_dir / img_info["file_name"].replace('.jpg', '.txt')
                
                # Move files if they exist
                if src_image_path.exists():
                    src_image_path.rename(dst_image_path)
                    moved_count += 1
                
                if src_label_path.exists():
                    src_label_path.rename(dst_label_path)
            
            print(f"   {split_name}: {moved_count} images moved to {split_name}/")
        
        # Remove empty original directories if they exist
        try:
            if self.images_dir.exists() and not any(self.images_dir.iterdir()):
                self.images_dir.rmdir()
            if self.labels_dir.exists() and not any(self.labels_dir.iterdir()):
                self.labels_dir.rmdir()
        except:
            pass

    def create_yolo_dataset_yaml(self, splits):
        """Create YOLO dataset configuration YAML"""
        print("Creating YOLO dataset configuration YAML...")
        
        # Create YOLO format configuration
        yolo_config = {
            'train': str(self.output_dir / 'train' / 'images'),
            'val': str(self.output_dir / 'val' / 'images'),
            'test': str(self.output_dir / 'test' / 'images'),
            'nc': len(self.yolo_classes),  # number of classes
            'names': self.yolo_classes  # class names
        }
        
        # Save YOLO dataset.yaml
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yolo_config, f, default_flow_style=False, indent=2)
        
        # Also save a metadata file with dataset information
        metadata_config = {
            'dataset_info': {
                'name': 'Video-derived YOLO Dataset',
                'description': 'Dataset created from video using RF-DETR auto-annotation',
                'total_images': len(self.yolo_data['images']),
                'total_annotations': len(self.yolo_data['annotations']),
                'categories': len(self.yolo_classes),
                'created_date': datetime.now().isoformat()
            },
            'splits': {
                'train': {
                    'images': len([img for img in self.yolo_data['images'] if img['id'] in splits['train']]),
                    'annotations': len([ann for ann in self.yolo_data['annotations'] if ann['image_id'] in splits['train']])
                },
                'val': {
                    'images': len([img for img in self.yolo_data['images'] if img['id'] in splits['val']]),
                    'annotations': len([ann for ann in self.yolo_data['annotations'] if ann['image_id'] in splits['val']])
                },
                'test': {
                    'images': len([img for img in self.yolo_data['images'] if img['id'] in splits['test']]),
                    'annotations': len([ann for ann in self.yolo_data['annotations'] if ann['image_id'] in splits['test']])
                }
            },
            'categories': {i: name for i, name in enumerate(self.yolo_classes)},
            'training_config': {
                'input_format': 'yolo',
                'target_format': 'yolo',
                'confidence_threshold': self.confidence_threshold,
                'extraction_fps': self.fps
            }
        }
        
        # Save metadata
        metadata_path = self.output_dir / 'dataset_metadata.yaml'
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata_config, f, default_flow_style=False, indent=2)
        
        print(f"   YOLO dataset configuration saved to: {yaml_path}")
        print(f"   Dataset metadata saved to: {metadata_path}")
        
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
                "total_images": len(self.yolo_data['images']),
                "total_annotations": len(self.yolo_data['annotations']),
                "unique_categories": len(self.metrics['category_counts']),
                "avg_annotations_per_image": len(self.yolo_data['annotations']) / len(self.yolo_data['images']) if self.yolo_data['images'] else 0,
                "extraction_fps": self.fps,
                "confidence_threshold": self.confidence_threshold
            },
            "performance_metrics": {
                "extraction_time_seconds": self.metrics['extraction_time'],
                "annotation_time_seconds": self.metrics['annotation_time'],
                "total_processing_time_seconds": self.metrics['extraction_time'] + self.metrics['annotation_time'],
                "extraction_speed_fps": len(self.yolo_data['images']) / self.metrics['extraction_time'] if self.metrics['extraction_time'] > 0 else 0,
                "annotation_speed_fps": len(self.yolo_data['images']) / self.metrics['annotation_time'] if self.metrics['annotation_time'] > 0 else 0
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
                        "percentage": (count / len(self.yolo_data['annotations'])) * 100
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
                    "frames_per_second_overall": len(self.yolo_data['images']) / (self.metrics['extraction_time'] + self.metrics['annotation_time']),
                    "memory_efficiency_mb_per_frame": (psutil.virtual_memory().used / (1024**2)) / len(self.yolo_data['images']) if self.yolo_data['images'] else 0
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
        
        # Step 5: Save YOLO annotations and organize splits
        self.save_yolo_annotations(splits)
        
        # Step 6: Create YOLO dataset YAML
        self.create_yolo_dataset_yaml(splits)
        
        # Step 7: Generate EDA plots
        self.generate_eda_plots()
        
        # Step 8: Generate comprehensive report
        self.generate_comprehensive_report()
        
        self.metrics['total_time'] = time.time() - total_start
        
        print(f"\nDataset creation completed successfully!")
        print(f"Output directory: {self.output_dir}")
        print(f"Total processing time: {self.metrics['total_time']:.2f} seconds")
        print(f"Total images: {len(self.yolo_data['images']):,}")
        print(f"Total annotations: {len(self.yolo_data['annotations']):,}")
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
    creator = YOLODatasetCreator(
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
