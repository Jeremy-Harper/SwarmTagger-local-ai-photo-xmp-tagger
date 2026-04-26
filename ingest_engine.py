***

# The Python Script (`ai_photo_ingest.py`)

```python
import os
import time
import subprocess
import multiprocessing
import numpy as np
from pathlib import Path
import tempfile
import gc

# --- LIBRARY IMPORTS ---
from PIL import Image
import pillow_heif

# --- CONFIGURATION ---
# >>> CHANGE THESE PATHS BEFORE RUNNING <<<
TARGET_DIRECTORY = r"C:\Path\To\Your\Photos"  
EXIFTOOL_PATH = r"C:\Path\To\exiftool.exe"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic"} 

# ====================================================================
# --- THE ULTIMATE SWARM CONFIGURATOR ---
# Tune these numbers based on your hardware's VRAM.
# Setting any of these to 0 is perfectly safe and dynamically handled.
# ====================================================================

# GPU 0 (Primary GPU - e.g., 24GB VRAM)
NUM_FACE_WORKERS_GPU0  = 0   
NUM_SCENE_WORKERS_GPU0 = 6   

# GPU 1 (Secondary GPU - e.g., 10GB VRAM)
NUM_FACE_WORKERS_GPU1  = 2   
NUM_SCENE_WORKERS_GPU1 = 2   

# DISK WRITER (CPU)
NUM_WRITERS = 1              # 1 is optimal for batching (prevents ExifTool DLL collisions)
WRITER_BATCH_SIZE = 10       # Safe batch limit for fast flushing
QUEUE_MAX_SIZE = 5000        # Deep buffer for task balancing
# ====================================================================

def preflight_checks():
    print("\n" + "="*50)
    print(" RUNNING GPU & EXIFTOOL WARM-UP ".center(50, "="))
    print("="*50)
    import torch
    if torch.cuda.is_available():
        print(f"[OK] PyTorch GPU: DETECTED ({torch.cuda.get_device_name(0)})")
    else:
        print("[WARNING] PyTorch GPU NOT DETECTED. Processing will be extremely slow.")
        
    if not os.path.exists(EXIFTOOL_PATH):
        print(f"[FATAL] ExifTool not found at {EXIFTOOL_PATH}")
        exit(1)
        
    print("[WAIT] Warming up ExifTool to unpack DLLs safely...")
    try:
        subprocess.run([EXIFTOOL_PATH, "-ver"], capture_output=True, check=True, creationflags=0x08000000)
        print("[OK] ExifTool is unpacked and ready.\n" + "="*50 + "\n")
    except Exception as e:
        print(f"[FATAL] ExifTool warm-up failed. Error: {e}")
        exit(1)
    time.sleep(2)

def read_image_safely(filepath):
    try: return Image.open(filepath).convert('RGB')
    except Exception: return None

# ==========================================
# SCENE WORKER LOGIC (Florence-2)
# ==========================================
def scene_worker_loop(worker_name, scene_q, write_q, gpu_scene_cnt, active_scene_workers, q_scene_cnt, q_write_cnt, latest_files, gpu_id):
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM
    try:
        model_id = "microsoft/Florence-2-base" 
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(f"cuda:{gpu_id}").eval()
    except Exception as e:
        latest_files['error'] = f"{worker_name} Load Error: {e}"
        return

    while True:
        data = scene_q.get()
        if data is None: break
        
        with q_scene_cnt.get_lock(): q_scene_cnt.value -= 1
        
        filepath, faces = data["filepath"], data["faces"]
        latest_files[worker_name] = filepath
        
        try:
            pil_img = read_image_safely(filepath)
            if pil_img is None: continue
            
            inputs = processor(text="<DETAILED_CAPTION>", images=pil_img, return_tensors="pt").to(f"cuda:{gpu_id}")
            with torch.no_grad():
                generated_ids = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=128)
            caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            write_q.put({"filepath": filepath, "faces": faces, "caption": caption})
            with q_write_cnt.get_lock(): q_write_cnt.value += 1
            with gpu_scene_cnt.get_lock(): gpu_scene_cnt.value += 1
                
        except Exception as e:
            latest_files['error'] = f"{worker_name} Error: {e}"

    with active_scene_workers.get_lock():
        active_scene_workers.value -= 1
        if active_scene_workers.value == 0:
            for _ in range(NUM_WRITERS): write_q.put(None)

def dedicated_scene_worker(gpu_id, worker_id, scene_q, write_q, gpu_scene_cnt, active_scene_workers, q_scene_cnt, q_write_cnt, latest_files):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    pillow_heif.register_heif_opener()
    
    # Internal CUDA isolation means the script sees it as cuda:0 even if it's physically GPU 1
    internal_cuda_id = 0 
    gpu_label = f"GPU_{gpu_id}"
    worker_name = f"{gpu_label}_Scene_{worker_id}"
    
    scene_worker_loop(worker_name, scene_q, write_q, gpu_scene_cnt, active_scene_workers, q_scene_cnt, q_write_cnt, latest_files, internal_cuda_id)

# ==========================================
# FACE WORKER LOGIC (InsightFace)
# ==========================================
def face_worker(gpu_id, worker_id, file_q, scene_q, write_q, gpu_face_cnt, gpu_scene_cnt, active_face_workers, active_scene_workers, q_file_cnt, q_scene_cnt, q_write_cnt, latest_files, total_scene_workers):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    pillow_heif.register_heif_opener()
    import cv2, torch, insightface
    
    internal_cuda_id = 0 
    gpu_label = f"GPU_{gpu_id}"
    worker_name = f"{gpu_label}_Face_{worker_id}"
    
    try:
        face_model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        face_model.prepare(ctx_id=internal_cuda_id, det_size=(640, 640))
    except Exception as e:
        latest_files['error'] = f"{worker_name} Load: {e}"
        return
        
    while True:
        filepath = file_q.get()
        if filepath is None: break
        
        with q_file_cnt.get_lock(): q_file_cnt.value -= 1
        latest_files[worker_name] = filepath
        
        try:
            pil_img = read_image_safely(filepath)
            if pil_img is None: continue
            
            img_cv2 = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            faces = face_model.get(img_cv2)
            face_names = [f"Person_Face_{i+1}" for i in range(len(faces))]
            
            scene_q.put({"filepath": filepath, "faces": face_names})
            with q_scene_cnt.get_lock(): q_scene_cnt.value += 1
            with gpu_face_cnt.get_lock(): gpu_face_cnt.value += 1
            
        except Exception as e:
            latest_files['error'] = f"{worker_name} Error: {e}"
            
    # When the last Face Worker finishes, signal all Dedicated Scene Workers to begin shutdown sequence
    with active_face_workers.get_lock():
        active_face_workers.value -= 1
        if active_face_workers.value == 0:
            for _ in range(total_scene_workers): scene_q.put(None)
                
    latest_files[worker_name] = "FINISHED FACES. MORPHING TO SCENE TAGGER..."
    del face_model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Morph into a scene worker to help empty the remaining scene queue
    scene_worker_loop(f"{gpu_label}_Scene_Morphed_{worker_id}", scene_q, write_q, gpu_scene_cnt, active_scene_workers, q_scene_cnt, q_write_cnt, latest_files, internal_cuda_id)

# ==========================================
# BATCH EXIFTOOL WRITER
# ==========================================
def exiftool_writer(write_q, write_count, q_write_cnt, latest_files):
    batch = []
    def process_batch(current_batch):
        if not current_batch: return
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8', suffix='.txt') as tmp:
            arg_file_path = tmp.name
            for item in current_batch:
                filepath, faces, caption = item["filepath"], item["faces"], item["caption"]
                xmp_path = str(Path(filepath).with_suffix('.xmp'))
                tmp.write(f"-overwrite_original\n-charset\nutf8\n")
                if not os.path.exists(xmp_path): tmp.write(f"-tagsfromfile\n{filepath}\n")
                tmp.write(f"-XMP:Description={caption}\n")
                if faces:
                    for face in faces: tmp.write(f"-XMP:PersonInImage+={face}\n")
                tmp.write(f"{xmp_path}\n-execute\n")
        try:
            subprocess.run([EXIFTOOL_PATH, "-@", arg_file_path], capture_output=True, text=True, check=True, creationflags=0x08000000)
            with write_count.get_lock(): write_count.value += len(current_batch)
        except subprocess.CalledProcessError as e:
            latest_files['error'] = f"ExifTool Batch Failed: {e.stderr[:150]}"
        finally:
            os.remove(arg_file_path)

    while True:
        data = write_q.get()
        if data is None:
            process_batch(batch)
            break
        with q_write_cnt.get_lock(): q_write_cnt.value -= 1
        latest_files['Writer'] = data["filepath"]
        batch.append(data)
        if len(batch) >= WRITER_BATCH_SIZE:
            process_batch(batch)
            batch = [] 

# ==========================================
# DASHBOARD
# ==========================================
def monitor_dashboard(gpu_0_face_cnt, gpu_0_scene_cnt, gpu_1_face_cnt, gpu_1_scene_cnt, write_count, skip_count, q_file_cnt, q_scene_cnt, q_write_cnt, latest_files, start_time):
    while True:
        time.sleep(10)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        if start_time.value == 0.0:
            elapsed, rate = 0, 0
        else:
            elapsed = time.time() - start_time.value
            rate = write_count.value / elapsed if elapsed > 0 else 0
            
        if q_file_cnt.value == 0 and start_time.value > 0: bottleneck = "Main Thread (Reading Drive)"
        elif q_scene_cnt.value < 10 and start_time.value > 0: bottleneck = "Face Mappers (Need more!)"
        elif q_write_cnt.value < 10 and start_time.value > 0: bottleneck = "Scene Taggers (GPUs Maxed)"
        else: bottleneck = "Healthy Flow"
        
        print("=" * 115)
        print(" PIPELINE INGESTION DASHBOARD (DYNAMIC SWARM) ".center(115, "="))
        print("=" * 115)
        print(f" TRUE TIME ELAPSED : {round(elapsed / 60, 2)} Minutes | TRUE SPEED: {round(rate, 2)} img/sec")
        print(f" TOTAL SKIPPED     : {skip_count.value} | TOTAL WRITTEN: {write_count.value}")
        print("-" * 115)
        print(f" STATUS       : {latest_files.get('status', 'Initializing...')}")
        print(f" BOTTLENECK   : {bottleneck}")
        print("-" * 115)
        print(" [QUEUES WAITING IN RAM] ")
        print(f"   -> Pending Faces: {q_file_cnt.value} | Pending Scenes: {q_scene_cnt.value}/{QUEUE_MAX_SIZE} | Pending Writes: {q_write_cnt.value}")
        print("-" * 115)
        print(f" [GPU 0] Primary   ({NUM_FACE_WORKERS_GPU0} Face / {NUM_SCENE_WORKERS_GPU0} Scene)")
        print(f"         Processed -> Faces: {gpu_0_face_cnt.value} | Scenes: {gpu_0_scene_cnt.value}")
        print(f"         Latest Action: {latest_files.get('GPU_0_Face_0', latest_files.get('GPU_0_Scene_0', '...'))}")
        print("-" * 115)
        print(f" [GPU 1] Secondary ({NUM_FACE_WORKERS_GPU1} Face / {NUM_SCENE_WORKERS_GPU1} Scene)")
        print(f"         Processed -> Faces: {gpu_1_face_cnt.value} | Scenes: {gpu_1_scene_cnt.value}")
        print(f"         Latest Action: {latest_files.get('GPU_1_Face_0', latest_files.get('GPU_1_Scene_0', '...'))}")
        print("-" * 115)
        print(f" [CPU] SSD WRITER | Written to disk: {write_count.value}")
        print(f"         Latest : {latest_files.get('Writer', '...')}")
        print("-" * 115)
        if latest_files.get('error'):
            print(f" LATEST ERROR: {latest_files['error']}")
            print("=" * 115)

if __name__ == '__main__':
    preflight_checks()
    manager = multiprocessing.Manager()
    latest_files = manager.dict({'error': ''})
    
    gpu_0_face_cnt, gpu_0_scene_cnt = multiprocessing.Value('i', 0), multiprocessing.Value('i', 0)
    gpu_1_face_cnt, gpu_1_scene_cnt = multiprocessing.Value('i', 0), multiprocessing.Value('i', 0)
    write_count, skip_count = multiprocessing.Value('i', 0), multiprocessing.Value('i', 0)
    q_file_cnt, q_scene_cnt, q_write_cnt = (multiprocessing.Value('i', 0) for _ in range(3))
    
    total_face_workers = NUM_FACE_WORKERS_GPU0 + NUM_FACE_WORKERS_GPU1
    total_scene_workers = NUM_SCENE_WORKERS_GPU0 + NUM_SCENE_WORKERS_GPU1 + total_face_workers
    
    active_face_workers = multiprocessing.Value('i', total_face_workers)
    active_scene_workers = multiprocessing.Value('i', total_scene_workers)
    start_time = multiprocessing.Value('d', 0.0)

    file_q = multiprocessing.Queue(maxsize=QUEUE_MAX_SIZE)
    scene_q = multiprocessing.Queue(maxsize=QUEUE_MAX_SIZE)
    write_q = multiprocessing.Queue(maxsize=QUEUE_MAX_SIZE)

    workers = []
    
    # Spawn GPU 0 Face Workers
    for i in range(NUM_FACE_WORKERS_GPU0):
        p = multiprocessing.Process(target=face_worker, args=(0, i, file_q, scene_q, write_q, gpu_0_face_cnt, gpu_0_scene_cnt, active_face_workers, active_scene_workers, q_file_cnt, q_scene_cnt, q_write_cnt, latest_files, total_scene_workers))
        p.start(); workers.append(p)
        
    # Spawn GPU 1 Face Workers
    for i in range(NUM_FACE_WORKERS_GPU1):
        p = multiprocessing.Process(target=face_worker, args=(1, i, file_q, scene_q, write_q, gpu_1_face_cnt, gpu_1_scene_cnt, active_face_workers, active_scene_workers, q_file_cnt, q_scene_cnt, q_write_cnt, latest_files, total_scene_workers))
        p.start(); workers.append(p)
        
    # Spawn GPU 0 Scene Workers
    for i in range(NUM_SCENE_WORKERS_GPU0):
        p = multiprocessing.Process(target=dedicated_scene_worker, args=(0, i, scene_q, write_q, gpu_0_scene_cnt, active_scene_workers, q_scene_cnt, q_write_cnt, latest_files))
        p.start(); workers.append(p)
        
    # Spawn GPU 1 Scene Workers
    for i in range(NUM_SCENE_WORKERS_GPU1):
        p = multiprocessing.Process(target=dedicated_scene_worker, args=(1, i, scene_q, write_q, gpu_1_scene_cnt, active_scene_workers, q_scene_cnt, q_write_cnt, latest_files))
        p.start(); workers.append(p)
        
    # Spawn CPU Writer
    for i in range(NUM_WRITERS):
        p = multiprocessing.Process(target=exiftool_writer, args=(write_q, write_count, q_write_cnt, latest_files))
        p.start(); workers.append(p)

    p_mon = multiprocessing.Process(target=monitor_dashboard, args=(gpu_0_face_cnt, gpu_0_scene_cnt, gpu_1_face_cnt, gpu_1_scene_cnt, write_count, skip_count, q_file_cnt, q_scene_cnt, q_write_cnt, latest_files, start_time))
    p_mon.daemon = True; p_mon.start()

    latest_files['status'] = 'Skipping existing files. Speed metrics will start when new files are found...'
    current_folder = ""
    
    for path in Path(TARGET_DIRECTORY).rglob('*'):
        if path.parent != current_folder:
            current_folder = path.parent
            latest_files['status'] = f"Scanning: ...\\{os.path.basename(current_folder)}"

        if path.suffix.lower() in VALID_EXTENSIONS:
            if path.with_suffix('.xmp').exists():
                with skip_count.get_lock(): skip_count.value += 1
                continue
            
            if start_time.value == 0.0:
                start_time.value = time.time()
                latest_files['status'] = 'Processing new files!'
                
            file_q.put(str(path))
            with q_file_cnt.get_lock(): q_file_cnt.value += 1

    latest_files['status'] = 'Disk scan complete. Workers are finishing the final items in the queues.'
    
    if total_face_workers > 0:
        for _ in range(total_face_workers): file_q.put(None)
    else:
        for _ in range(total_scene_workers): scene_q.put(None)
        
    for w in workers: w.join()
    print("\nPipeline Complete.")
