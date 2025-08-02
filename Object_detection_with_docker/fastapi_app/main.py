from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from pydantic import BaseModel, validator
import os, tempfile, glob, traceback, cv2, time, numpy as np
import torch

# 🔐 PyTorch safe loading (for v2.6+)
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel
add_safe_globals([DetectionModel])  # Trust YOLO class

# ⛓️ Redis utilities
from redis_cache import get_cache, set_cache, purge_cache

app = FastAPI()

@app.on_event("startup")
def startup_event():
    try:
        print("♻️ Purging Redis cache on startup...")
        purge_cache()
    except Exception as e:
        print("⚠️ Redis purge failed:", e)

# Override default weights_only behavior globally
_original_load = torch.load

def trusted_load(*args, **kwargs):
    kwargs["weights_only"] = False  # Force full deserialization
    return _original_load(*args, **kwargs)

torch.load = trusted_load

# 🧠 Load YOLO model (safe path)
model_path = os.path.join(os.getcwd(), "best.pt")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")
model = YOLO(model_path)

# 📁 Output directories
DETECT_DIR = os.path.abspath("outputs/images")
TRACK_DIR = os.path.abspath("outputs/tracked_videos")
os.makedirs(DETECT_DIR, exist_ok=True)
os.makedirs(TRACK_DIR, exist_ok=True)

def get_color(class_name: str):
    np.random.seed(hash(class_name) % (2**32))
    return tuple(np.random.randint(0, 256, 3).tolist())

tracker_config = "bytetrack.yaml"

VALID_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
VALID_VIDEO_TYPES = {"video/mp4", "video/x-matroska", "video/avi", "video/mov", "video/quicktime"}

class DetectionInput(BaseModel):
    confidence: float = 0.15
    @validator("confidence")
    def validate_conf(cls, v):
        if not (0.0 < v <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v

class VideoInput(BaseModel):
    confidence: float = 0.15
    @validator("confidence")
    def validate_conf(cls, v):
        if not (0.0 < v <= 1.0):
            raise ValueError("Confidence must be between 0 and 1.")
        return v

@app.get("/")
def root():
    return {"message": "Server is live!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/detect")
async def detect_image(file: UploadFile = File(...), confidence: float = Form(0.15)):
    try:
        input_data = DetectionInput(confidence=confidence)

        if file.content_type not in VALID_IMAGE_TYPES:
            raise HTTPException(status_code=400, detail=f"Unsupported image type: {file.content_type}")

        original_name = os.path.splitext(file.filename)[0]
        cache_key = f"detect_{original_name}_{input_data.confidence}"
        cached_path = get_cache(cache_key)

        if cached_path and os.path.exists(cached_path):
            print("📦 Returning cached result:", cached_path)
            img_file = open(cached_path, "rb")
            headers = {"X-Inference-Time-in-seconds": "0.0"}
            return StreamingResponse(img_file, media_type="image/jpeg", headers=headers)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
            tmp_img.write(await file.read())
            image_path = tmp_img.name

        print(f"🔍 Detecting: {file.filename}")
        print(f"🎛 Confidence: {input_data.confidence}")
        start_time = time.time()

        result = model.predict(image_path)[0]
        boxes = getattr(result.boxes, "data", None)
        if boxes is None:
            raise ValueError("No bounding boxes detected.")

        boxes_np = boxes.cpu().numpy()
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image read error.")

        for box in boxes_np:
            if len(box) == 6:
                x1, y1, x2, y2, conf, cls = box
                if conf < input_data.confidence:
                    continue
                cls_id = int(cls)
                name = model.names.get(cls_id, "Unknown")
                label = f"{name} {conf:.2f}"
                color = get_color(name)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(img, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        output_path = os.path.join(DETECT_DIR, f"{original_name}_detected.jpg")
        if not cv2.imwrite(output_path, img):
            raise IOError("Failed to save image.")
        os.remove(image_path)

        set_cache(cache_key, output_path)
        inference_time = round(time.time() - start_time, 3)
        print(f"📸 Saved: {output_path}")
        img_file = open(output_path, "rb")
        headers = {
            "Content-Disposition": f'inline; filename="{os.path.basename(output_path)}"',
            "X-Inference-Time-in-seconds": str(inference_time)
        }
        return StreamingResponse(img_file, media_type="image/jpeg", headers=headers)

    except Exception as e:
        print("❌ Detect error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/track_video")
async def track_video(file: UploadFile = File(...), confidence: float = Form(0.15)):
    try:
        input_data = VideoInput(confidence=confidence)

        if file.content_type not in VALID_VIDEO_TYPES:
            raise HTTPException(status_code=400, detail=f"Unsupported video type: {file.content_type}")

        original_name = os.path.splitext(file.filename)[0]
        cache_key = f"track_{original_name}_{input_data.confidence}"
        cached_video_path = get_cache(cache_key)

        if cached_video_path and os.path.exists(cached_video_path):
            print("📦 Returning cached tracked video:", cached_video_path)
            video_file = open(cached_video_path, "rb")
            headers = {
                "Content-Disposition": f'attachment; filename="{os.path.basename(cached_video_path)}"',
                "X-Inference-Time-in-seconds": "0.0"
            }
            return StreamingResponse(video_file, media_type="video/mp4", headers=headers)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(await file.read())
            input_path = tmp_video.name

        output_folder = os.path.join(TRACK_DIR, f"{original_name}_tracked")
        print(f"🎬 Tracking: {file.filename}")
        print(f"🎛 Confidence: {input_data.confidence}")
        start_time = time.time()

        _ = model.track(
            source=input_path,
            conf=input_data.confidence,
            iou=0.5,
            persist=True,
            show=False,
            tracker=tracker_config,
            save=True,
            project=TRACK_DIR,
            name=f"{original_name}_tracked"
        )
        inference_time = round(time.time() - start_time, 3)

        tracked_files = glob.glob(os.path.join(output_folder, "*.*"))
        tracked_files = [f for f in tracked_files if f.endswith((".mp4", ".avi", ".mov"))]

        tracked_files.sort(key=os.path.getmtime, reverse=True)
        final_path = tracked_files[0] if tracked_files else None

        print(f"output_folder: {output_folder} ")
        print(f"final_path is {final_path}")
	
        if not final_path or not os.path.exists(final_path):
            raise FileNotFoundError("Tracked video not found.")

        set_cache(cache_key, final_path)
        print(f"📦 Found: {final_path}")
        video_file = open(final_path, "rb")
        headers = {
            "Content-Disposition": f'attachment; filename="{os.path.basename(final_path)}"',
            "X-Inference-Time-in-seconds": str(inference_time)
        }
        return StreamingResponse(video_file, media_type="video/mp4", headers=headers)

    except Exception as e:
        print("❌ Track error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
