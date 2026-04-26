# SwarmTagger-local-ai-photo-xmp-tagger
Multi-GPU Swarm to build XMP faces/description as fast as possible
Its a python AI Photo Tagger script to saturate my Dual-GPU Hybrid Swarm

A high-performance, multi-threaded Python pipeline that leverages local AI models to automatically detect faces and generate descriptive captions for massive photo libraries. 

By writing metadata directly to standard `.xmp` sidecar files, this tool perfectly prepares your library for open-source image management systems (like digiKam, Immich, PhotoPrism, or Lightroom) without modifying your original images.

## The Problem
When dealing with massive photo libraries (100,000 to 1,000,000+ images), standard open-source workflows hit a wall. Tools like digiKam are fantastic, but relying on them to sequentially process face detection and scene classification on hundreds of thousands of images can take **60+ days** of non-stop CPU/GPU computing due to a lack of deep multi-threading and modern VRAM management.

## The Solution
This script utilizes a **Hybrid Swarm Architecture** designed to push multi-GPU consumer systems to their theoretical maximum throughput. 
* **Speed:** By batching disk I/O and running multiple AI models in parallel across isolated GPUs, this pipeline can process **3 to 6 images per second** (reducing a 60-day workload to less than a weekend).
* **Phase Morphing:** GPU workers dynamically switch from Face Mapping to Scene Tagging when the queue empties to ensure 100% hardware utilization.
* **Non-Destructive:** Writes directly to `.xmp` sidecars using a persistent ExifTool batch writer, leaving your original `.JPG` and `.HEIC` files completely untouched.
* **Smart Resume:** Instantly detects existing `.xmp` files and resumes where it left off in seconds.

## AI Models Used
* **Faces:** `InsightFace` (buffalo_l) - Extremely fast, highly accurate CNN face mapping.
* **Scenes:** `Microsoft Florence-2-base` - State-of-the-art vision-language model for incredibly detailed caption generation.

---

## Installation & Prerequisites

**1. System Requirements**
* Windows / Linux
* Python 3.10+
* At least one NVIDIA GPU (CUDA enabled). 
* [ExifTool](https://exiftool.org/) installed or downloaded.

**2. Install Dependencies**
To ensure the models utilize your CUDA cores (and not your CPU), install the specific PyTorch and ONNX versions:
```bash
pip uninstall -y torch torchvision torchaudio onnxruntime onnxruntime-gpu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-gpu insightface transformers timm==0.9.12 einops pillow pillow-heif opencv-python




Here is an example of the terminal output which refreshes every ~10 seconds

===================================================================================================================
=================================== PIPELINE INGESTION DASHBOARD (DYNAMIC SWARM) ==================================
===================================================================================================================
 TRUE TIME ELAPSED : 74.53 Minutes | TRUE SPEED: 4.79 img/sec
 TOTAL SKIPPED     : 134715 | TOTAL WRITTEN: 21400
-------------------------------------------------------------------------------------------------------------------
 STATUS       : Scanning: ...\A
 BOTTLENECK   : Scene Taggers (GPUs Maxed)
-------------------------------------------------------------------------------------------------------------------
 [QUEUES WAITING IN RAM]
   -> Pending Faces: 5000 | Pending Scenes: 2947/5000 | Pending Writes: 0
-------------------------------------------------------------------------------------------------------------------
 [GPU 0] RTX 3090 (0 Face / 6 Scene)
         Processed -> Faces: 0 | Scenes: 13522
         Latest Action: F:\2020\Photos\0ass0asaass.heic
-------------------------------------------------------------------------------------------------------------------
 [GPU 1] RTX 3080 (2 Face / 2 Scene)
         Processed -> Faces: 24364 | Scenes: 7887
         Latest Action: F:\2020\Photos\asasasasasa.heic
-------------------------------------------------------------------------------------------------------------------
 [CPU] SSD WRITER | Written to disk: 21400
         Latest : F:\2020\Photos\asasaasasa.heic
-------------------------------------------------------------------------------------------------------------------


