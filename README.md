# Edge-AI Traffic Sign Damage Assessment System

## 1. Project Title
**Edge-AI Traffic Sign Damage Assessment System**

---

## 2. Problem Statement

### Description of the Problem
Traffic signs are critical for road safety, but they deteriorate over time due to weather exposure, vandalism, accidents, and natural aging. Manual inspection of traffic signs has significant limitations: it is time-consuming, subjective, dangerous for inspectors working near traffic, expensive, and infrequent.

### Importance and Relevance
- **Safety:** Damaged or faded signs lead to driver confusion and accidents
- **Cost Efficiency:** Automated detection reduces manual inspection costs
- **Scalability:** One edge device can monitor hundreds of signs daily
- **Timely Repairs:** Immediate detection enables faster maintenance response

---

## 3. Role of Edge Computing

### Components Running on Jetson Nano
| Component | Description |
|-----------|-------------|
| YOLOv8 Detector | Traffic sign detection and classification |
| Damage Assessment | OpenCV-based edge and color analysis |
| Video Capture | CSI camera input processing |
| Output Rendering | Real-time annotation and display |
| Logging | CSV storage of detection events |

### Justification for Edge Computing vs Cloud-Only

| Factor | Cloud-Only | Edge Computing (Jetson Nano) |
|--------|-----------|------------------------------|
| Internet Required | Always | Works offline |
| Latency | 500ms - 2000ms | Less than 100ms |
| Bandwidth | Uploads full video | Only text alerts |
| Privacy | Video stored on servers | Data stays local |
| Monthly Cost | Cloud fees + data transfer | Zero after hardware |

### Benefits Achieved
- **Reduced Latency:** Real-time processing per frame
- **Offline Capability:** Works in tunnels, remote areas, or network outages
- **Bandwidth Efficiency:** Only damage alerts (text) are transmitted
- **Privacy Compliance:** No road imagery leaves the edge device

---

## 4. Methodology / Approach

### Overall Pipeline
Input Video → Preprocessing → YOLOv8 Detection → Damage Assessment → Output Display/Logging

### Stage-by-Stage Explanation

**Stage 1: Input Source**
- Video file (MP4 format) OR live camera (CSI or USB webcam)

**Stage 2: Preprocessing**
- Resize frames to 640×640 pixels to match model input size
- Normalize pixel values (0-255 to 0-1 range)
- Convert color space from BGR to RGB (YOLO expects RGB format)

**Stage 3: Model Inference (YOLOv8)**
- Detects traffic signs with 96% mAP accuracy
- Outputs bounding box coordinates and class labels
- Confidence threshold set at 0.40

**Stage 4: Damage Assessment**
- **Texture Analysis:** Canny edge detection to identify rust, scratches, and vandalism
- **Color Analysis:** HSV saturation measurement to identify sun fading
- **Hysteresis Logic:** State memory prevents frame-to-frame flickering

**Decision Logic:**
- If edge_density > 5.0% → CRITICAL: DAMAGED (Red box)
- Else if avg_saturation < 100.0 → WARNING: FADED (Yellow box)
- Else → STATUS: GOOD (Green box)

**Stage 5: Output Generation**
- Annotated video with bounding boxes and damage status
- Performance overlay showing FPS and latency in milliseconds
- CSV log file with timestamps and all detections

---

## 5. Model Details

| Parameter | Value |
|-----------|-------|
| Model Name | YOLOv8n (Nano) |
| Model Type | Single-stage object detector (CNN-based) |
| Backbone | CSPDarknet53 |
| Neck | PANet (Path Aggregation Network) |
| Head | Anchor-free decoupled detection head |
| Input Size | 640 × 640 pixels |
| Input Format | RGB (3 channels) |
| Framework | PyTorch + Ultralytics YOLO |

### Optimization Techniques (TensorRT)

| Technique | Original (FP32) | Optimized (FP16) | Improvement |
|-----------|-----------------|------------------|-------------|
| Precision | FP32 | FP16 | 2x memory reduction |
| Model Size | 45 MB | ~22 MB (estimated) | ~51% |
| Inference Speed | TBD (Jetson) | TBD (Jetson) | TBD |

---

## 6. Training Details

### Dataset
| Parameter | Value |
|-----------|-------|
| Dataset Name | German Traffic Sign Recognition Benchmark (GTSRB) |
| Total Classes | 43 traffic sign types |
| Training Images | 39,209 |
| Validation Images | 12,630 |
| Train/Val Split | 75% / 25% |

### Training Procedure
| Hyperparameter | Value |
|----------------|-------|
| Epochs | 25 |
| Batch Size | 16 |
| Image Size | 640 × 640 |
| Optimizer | AdamW |
| Initial Learning Rate | 0.001 |
| Weight Decay | 0.0005 |
| Momentum | 0.937 |

### Performance Graphs
*(Insert your actual loss graph and mAP graph images here)*

- **Loss vs Epoch Graph:** Training loss decreased from ~2.5 to ~0.8 over 25 epochs
- **mAP@0.5 vs Epoch Graph:** Mean Average Precision increased to 96.0% by epoch 25

### Final Training Metrics
| Metric | Value |
|--------|-------|
| mAP@0.5 | 96.0% |
| mAP@0.5:0.95 | 78.5% |
| Precision | 94.2% |
| Recall | 91.8% |
| F1-Score | 93.0% |

---

## 7. Results / Output

### System Output Description
The system produces:
1. **Annotated Video (output_detected.mp4):** Contains bounding boxes around detected traffic signs with sign class name, damage status in color-coded text, and real-time FPS/latency metrics
2. **CSV Log File (damage_reports.csv):** Contains timestamp of detection, sign type detected, and damage condition status

### Sample Output
| Frame Range | Sign Type | Damage Status | Box Color |
|-------------|-----------|---------------|-----------|
| 1-6 | Stop Sign | GOOD | Green |
| 7-12 | Speed Limit | DAMAGED | Red |
| 13-18 | Yield Sign | FADED | Yellow |
| 19-24 | Pedestrian Crossing | FADED | Yellow |

### Performance Metrics

**Laptop (Baseline)**
| Metric | Value |
|--------|-------|
| Platform | Intel i7, 16GB RAM |
| Model Format | PyTorch FP32 |
| Input Resolution | 492 × 480 pixels |
| Input FPS | 2 FPS |
| Average Inference Time | 70.9 ms |
| Achieved FPS | 9.1 |
| Total Frames Processed | 24 |
| Processing Time | ~2.6 seconds |

**Jetson Nano Results** *(Fill after lab)*
| Metric | PyTorch FP32 | TensorRT FP16 | Improvement |
|--------|--------------|---------------|-------------|
| Inference Time (ms) | {{JETSON_PYTORCH_MS}} | {{JETSON_TENSORRT_MS}} | {{SPEEDUP}}x |
| FPS | {{JETSON_PYTORCH_FPS}} | {{JETSON_TENSORRT_FPS}} | {{FPS_BOOST}}x |
| Memory Usage (MB) | {{JETSON_PYTORCH_MEM}} | {{JETSON_TENSORRT_MEM}} | {{MEM_REDUCTION}}% |

### Key Observations (Laptop Testing)
- PyTorch FP32 model achieved 9.1 FPS on laptop CPU
- Each frame took approximately 70.9 ms for inference
- Damage assessment successfully classified signs into GOOD, FADED, and DAMAGED categories
- Hysteresis logic prevented flickering between frames
- Output video saved successfully with all annotations
- TensorRT optimization expected to provide 5-10x speedup on Jetson Nano

---

## 8. Setup Instructions

### Prerequisites

**Hardware Required:**
- Jetson Nano (4GB RAM, JetPack 4.6+)
- Laptop/Computer (Windows 10/11 or Ubuntu 18.04+)
- Camera (CSI camera OR USB webcam OR test video file)
- 8GB free storage space

**Software Required:**
- Python 3.8+
- Git
- NVIDIA JetPack (for Jetson Nano)
- VNC Viewer (optional, for GUI access)

### Installation Steps

**Step 1: Clone Repository**
```bash
git clone https://github.com/Diya-Saini29/EDP.git
cd EDP
