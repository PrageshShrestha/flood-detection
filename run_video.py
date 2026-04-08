from ultralytics import YOLO
import cv2
import time
import psutil
import threading
import numpy as np
from collections import deque
import GPUtil
import torch

class YOLOResourceTracker:
    def __init__(self, model_path='best.onnx', device='cpu'):
        self.model = YOLO(model_path)
        self.device = device
        
        # Resource tracking variables
        self.cpu_percents = []
        self.ram_usages = []
        self.gpu_usage = []
        self.vram_usage = []
        self.fps_values = deque(maxlen=30)  # Last 30 FPS values
        self.inference_times = []
        self.frame_times = []
        
        # Performance metrics for current run
        self.total_frames = 0
        self.total_detections = 0
        self.confidence_scores = []
        
        # Thread safety
        self.tracking = False
        self.resource_thread = None
        
    def start_resource_monitoring(self):
        """Start background thread to monitor system resources"""
        self.tracking = True
        self.resource_thread = threading.Thread(target=self._monitor_resources)
        self.resource_thread.daemon = True
        self.resource_thread.start()
        
    def _monitor_resources(self):
        """Monitor CPU, RAM, GPU in background"""
        while self.tracking:
            # CPU Usage
            self.cpu_percents.append(psutil.cpu_percent(interval=0.1))
            
            # RAM Usage (GB)
            ram = psutil.virtual_memory()
            self.ram_usages.append(ram.used / (1024**3))
            
            # GPU Monitoring (if available and device is cuda)
            if self.device == 'cuda' or torch.cuda.is_available():
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        self.gpu_usage.append(gpus[0].load * 100)
                        self.vram_usage.append(gpus[0].memoryUsed / 1024)  # GB
                except:
                    pass
            
            time.sleep(0.1)  # Sample every 100ms
    
    def stop_resource_monitoring(self):
        """Stop resource monitoring"""
        self.tracking = False
        if self.resource_thread:
            self.resource_thread.join(timeout=1)
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def process_video(self, video_path='test.mp4', ground_truth=None):
        """Process video with full resource and performance tracking"""
        
        # Start monitoring
        self.start_resource_monitoring()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        fps_in = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        
        # Metrics storage
        precision_vals = []
        recall_vals = []
        iou_vals = []
        
        print("=" * 80)
        print("YOLO PERFORMANCE & RESOURCE TRACKING")
        print("=" * 80)
        print(f"{'Frame':<8} {'FPS':<8} {'CPU%':<8} {'RAM(GB)':<10} {'GPU%':<8} {'VRAM(GB)':<10} {'Detections':<12} {'Inference(ms)':<15}")
        print("-" * 80)
        
        # Inference with stream=True for memory efficiency
        results = self.model.predict(
            source=video_path,
            stream=True,
            device=self.device,
            verbose=False
        )
        
        for result in results:
            frame_start = time.time()
            
            # Get detections
            boxes = result.boxes
            num_detections = 0
            frame_confidences = []
            
            if boxes is not None:
                num_detections = len(boxes)
                self.total_detections += num_detections
                
                # Collect confidence scores
                if boxes.conf is not None:
                    confs = boxes.conf.cpu().numpy()
                    frame_confidences.extend(confs)
                    self.confidence_scores.extend(confs)
            
            # Calculate inference time (from result object if available)
            if hasattr(result, 'speed'):
                inference_ms = result.speed.get('inference', 0)
            else:
                inference_ms = (time.time() - frame_start) * 1000
            
            self.inference_times.append(inference_ms)
            
            # Calculate current FPS
            frame_time = time.time() - frame_start
            self.frame_times.append(frame_time)
            current_fps = 1.0 / frame_time if frame_time > 0 else 0
            self.fps_values.append(current_fps)
            
            # Get current resource stats
            cpu_avg = np.mean(self.cpu_percents[-10:]) if self.cpu_percents else 0
            ram_avg = np.mean(self.ram_usages[-10:]) if self.ram_usages else 0
            gpu_avg = np.mean(self.gpu_usage[-10:]) if self.gpu_usage else 0
            vram_avg = np.mean(self.vram_usage[-10:]) if self.vram_usage else 0
            
            # Print real-time stats
            print(f"{frame_count:<8} {current_fps:<8.1f} {cpu_avg:<8.1f} {ram_avg:<10.2f} "
                  f"{gpu_avg:<8.1f} {vram_avg:<10.2f} {num_detections:<12} {inference_ms:<15.2f}")
            
            # Display frame (optional, can be disabled for pure tracking)
            if result.orig_img is not None:
                # Annotate frame
                annotated = result.plot()
                cv2.imshow('YOLO Tracking', annotated)
                if cv2.waitKey(1) == ord('q'):
                    break
            
            frame_count += 1
            self.total_frames = frame_count
        
        # Stop monitoring
        self.stop_resource_monitoring()
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final report
        self._print_final_report()
        
    def _print_final_report(self):
        """Generate and print comprehensive performance report"""
        print("\n" + "=" * 80)
        print("FINAL PERFORMANCE & RESOURCE REPORT")
        print("=" * 80)
        
        # Inference Performance
        print("\n📊 INFERENCE PERFORMANCE:")
        print(f"   Total Frames Processed: {self.total_frames}")
        print(f"   Total Detections: {self.total_detections}")
        print(f"   Average FPS: {np.mean(self.fps_values):.2f}")
        print(f"   FPS Range: {min(self.fps_values):.2f} - {max(self.fps_values):.2f}")
        print(f"   Average Inference Time: {np.mean(self.inference_times):.2f} ms")
        print(f"   Inference Time Std Dev: {np.std(self.inference_times):.2f} ms")
        print(f"   P95 Inference Time: {np.percentile(self.inference_times, 95):.2f} ms")
        
        # System Resources
        print("\n💻 SYSTEM RESOURCES:")
        print(f"   Average CPU Usage: {np.mean(self.cpu_percents):.1f}%")
        print(f"   Peak CPU Usage: {max(self.cpu_percents):.1f}%")
        print(f"   Average RAM Usage: {np.mean(self.ram_usages):.2f} GB")
        print(f"   Peak RAM Usage: {max(self.ram_usages):.2f} GB")
        
        if self.gpu_usage:
            print(f"   Average GPU Usage: {np.mean(self.gpu_usage):.1f}%")
            print(f"   Peak GPU Usage: {max(self.gpu_usage):.1f}%")
            print(f"   Average VRAM Usage: {np.mean(self.vram_usage):.2f} GB")
            print(f"   Peak VRAM Usage: {max(self.vram_usage):.2f} GB")
        
        # Detection Quality (based on confidence scores)
        if self.confidence_scores:
            print("\n🎯 DETECTION QUALITY:")
            print(f"   Average Confidence: {np.mean(self.confidence_scores):.3f}")
            print(f"   Median Confidence: {np.median(self.confidence_scores):.3f}")
            print(f"   Confidence Std Dev: {np.std(self.confidence_scores):.3f}")
            
            # Distribution of confidences
            high_conf = sum(1 for c in self.confidence_scores if c > 0.7)
            med_conf = sum(1 for c in self.confidence_scores if 0.5 <= c <= 0.7)
            low_conf = sum(1 for c in self.confidence_scores if c < 0.5)
            print(f"   High Confidence (>0.7): {high_conf} ({high_conf/len(self.confidence_scores)*100:.1f}%)")
            print(f"   Medium Confidence (0.5-0.7): {med_conf} ({med_conf/len(self.confidence_scores)*100:.1f}%)")
            print(f"   Low Confidence (<0.5): {low_conf} ({low_conf/len(self.confidence_scores)*100:.1f}%)")
        
        # Resource Efficiency Score (custom metric)
        avg_fps = np.mean(self.fps_values)
        avg_cpu = np.mean(self.cpu_percents)
        efficiency_score = (avg_fps / avg_cpu) * 100 if avg_cpu > 0 else 0
        print(f"\n⚡ RESOURCE EFFICIENCY SCORE: {efficiency_score:.2f} (FPS per %CPU)")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if avg_fps < 15:
            print("   ⚠️  Low FPS detected. Consider: reducing input size, using TensorRT, or better hardware")
        if np.mean(self.inference_times) > 100:
            print("   ⚠️  High inference latency. Consider model optimization or quantization")
        if np.mean(self.cpu_percents) > 80:
            print("   ⚠️  High CPU usage. Consider using GPU or reducing batch operations")
        if self.confidence_scores and np.std(self.confidence_scores) > 0.3:
            print("   ℹ️  High confidence variance. Consider adjusting confidence threshold")
        
        print("\n" + "=" * 80)


# Usage
if __name__ == "__main__":
    # Install required packages if missing:
    # pip install psutil gputil
    
    # Initialize tracker
    tracker = YOLOResourceTracker(
        model_path='best.onnx',
        device='cpu'  # or 'cuda' for GPU
    )
    
    # Process video with full tracking
    tracker.process_video('test.mp4')