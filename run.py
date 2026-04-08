import time
import psutil
import os
from ultralytics import YOLO

# ---- Process tracker (IMPORTANT) ----
process = psutil.Process(os.getpid())

def get_ram_gb():
    return process.memory_info().rss / (1024**3)

# ---- BEFORE loading model ----
ram_before = get_ram_gb()

# ---- Load ONNX model ----
model = YOLO("best.onnx")

# ---- AFTER loading model ----
ram_after = get_ram_gb()

print("\n==============================")
print(f"Model RAM Usage: {ram_after - ram_before:.3f} GB")
print("==============================\n")

max_ram = 0

# Warm-up run (important for fair FPS)
_ = model("image.png", device="cpu")

# ---- Benchmark loop ----
for i in range(5):

    t0 = time.time()

    results = model("image.png", device="cpu")

    t1 = time.time()

    # ---- RAM (process only) ----
    ram_used = get_ram_gb()
    max_ram = max(max_ram, ram_used)

    # ---- CPU usage ----
    cpu = psutil.cpu_percent(interval=0.1)

    # ---- Timing ----
    inference_time = t1 - t0
    fps = 1 / inference_time if inference_time > 0 else 0

    print(f"\n--- Run {i+1} ---")
    print(f"Inference time: {inference_time:.3f} sec")
    print(f"FPS: {fps:.2f}")
    print(f"CPU Usage: {cpu:.1f}%")
    print(f"Process RAM: {ram_used:.3f} GB")

# ---- Final summary ----
print("\n==============================")
print(f"🔥 Peak Process RAM: {max_ram:.3f} GB")
print("==============================\n")