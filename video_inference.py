import cv2
import time
import psutil
import torch
import pandas as pd
from ultralytics import YOLO
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "best.pt"
VIDEO_PATH = "input.mp4"
OUTPUT_VIDEO = "output.mp4"
METRICS_CSV = "metrics_log.csv"
SUMMARY_FILE = "summary.txt"

CONF_THRESHOLD = 0.25

# -----------------------------
# LOAD MODEL
# -----------------------------
device = "cpu"  # Raspberry Pi 3 = CPU only
model = YOLO(MODEL_PATH)

# -----------------------------
# VIDEO SETUP
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps_input, (width, height))

# -----------------------------
# METRICS STORAGE
# -----------------------------
logs = []

frame_count = 0
start_time = time.time()

cpu_usages = []
ram_usages = []
latencies = []

# -----------------------------
# PROCESS LOOP
# -----------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_start = time.time()

    # -----------------------------
    # SYSTEM METRICS BEFORE INFERENCE
    # -----------------------------
    cpu_before = psutil.cpu_percent(interval=None)
    ram_before = psutil.virtual_memory().percent

    # -----------------------------
    # YOLO INFERENCE
    # -----------------------------
    results = model(frame, conf=CONF_THRESHOLD, device=device)

    # Draw detections
    annotated_frame = results[0].plot()

    # -----------------------------
    # SYSTEM METRICS AFTER INFERENCE
    # -----------------------------
    cpu_after = psutil.cpu_percent(interval=None)
    ram_after = psutil.virtual_memory().percent

    frame_end = time.time()

    latency = frame_end - frame_start
    fps = 1.0 / latency if latency > 0 else 0

    # Store metrics
    cpu_usages.append(cpu_after)
    ram_usages.append(ram_after)
    latencies.append(latency)

    logs.append({
        "frame": frame_count,
        "cpu_percent": cpu_after,
        "ram_percent": ram_after,
        "latency_sec": latency,
        "fps": fps,
        "timestamp": datetime.now().strftime("%H:%M:%S.%f")
    })

    # Write frame
    out.write(annotated_frame)

    frame_count += 1

# -----------------------------
# FINALIZE
# -----------------------------
cap.release()
out.release()

end_time = time.time()
total_time = end_time - start_time

# -----------------------------
# SAVE METRICS
# -----------------------------
df = pd.DataFrame(logs)
df.to_csv(METRICS_CSV, index=False)

# -----------------------------
# SUMMARY STATISTICS
# -----------------------------
avg_cpu = sum(cpu_usages) / len(cpu_usages)
avg_ram = sum(ram_usages) / len(ram_usages)
avg_latency = sum(latencies) / len(latencies)
avg_fps = 1.0 / avg_latency if avg_latency > 0 else 0

with open(SUMMARY_FILE, "w") as f:
    f.write("===== YOLO Inference Performance Summary =====\n")
    f.write(f"Total Frames: {frame_count}\n")
    f.write(f"Total Time: {total_time:.2f} sec\n")
    f.write(f"Average CPU Usage: {avg_cpu:.2f}%\n")
    f.write(f"Average RAM Usage: {avg_ram:.2f}%\n")
    f.write(f"Average Latency per Frame: {avg_latency:.4f} sec\n")
    f.write(f"Average FPS: {avg_fps:.2f}\n")

print("✅ Done!")
print(f"📊 Metrics saved to: {METRICS_CSV}")
print(f"📄 Summary saved to: {SUMMARY_FILE}")
print(f"🎥 Output video saved to: {OUTPUT_VIDEO}")
