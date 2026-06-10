import json
import numpy as np
from pathlib import Path
from PIL import Image
import io

import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# ── Config ──────────────────────────────────────────────
MODEL_PATH       = Path("mobilenet_tuned.keras")
CLASS_NAMES_PATH = Path("class_names.json")
IMG_SIZE         = (128, 128)

# ── Load model & class names once at startup ─────────────
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH) as f:
    class_names = json.load(f)

print(f"Model loaded ✓  ({len(class_names)} classes)")

# ── FastAPI app ──────────────────────────────────────────
app = FastAPI(
    title="Plant Disease Detection API",
    description="Upload a leaf image and get the predicted disease class.",
    version="1.0.0"
)

# ── Helper ───────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # shape: (1, 128, 128, 3)

def parse_class_name(raw: str) -> dict:
    parts = raw.split("___")
    crop    = parts[0].replace("_", " ").strip()
    disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Unknown"
    return {"crop": crop, "disease": disease}

# ── Routes ───────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Plant Disease Detection API is running!",
        "usage":   "POST /predict with an image file",
        "classes": len(class_names)
    }

@app.get("/classes")
def get_classes():
    parsed = [parse_class_name(c) for c in class_names]
    return {"total": len(class_names), "classes": parsed}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400,
                            detail="File must be an image (jpg, png, etc.)")

    # Read & preprocess
    image_bytes = await file.read()
    try:
        img_array = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f"Could not process image: {str(e)}")

    # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_idx  = int(np.argmax(predictions[0]))
    confidence     = float(np.max(predictions[0]))

    # Top 3 predictions
    top3_idx = np.argsort(predictions[0])[::-1][:3]
    top3 = [
        {
            "rank":       int(i+1),
            "class":      class_names[idx],
            "crop":       parse_class_name(class_names[idx])["crop"],
            "disease":    parse_class_name(class_names[idx])["disease"],
            "confidence": round(float(predictions[0][idx]) * 100, 2)
        }
        for i, idx in enumerate(top3_idx)
    ]

    return JSONResponse({
        "filename":   file.filename,
        "prediction": {
            "class":      class_names[predicted_idx],
            "crop":       parse_class_name(class_names[predicted_idx])["crop"],
            "disease":    parse_class_name(class_names[predicted_idx])["disease"],
            "confidence": round(confidence * 100, 2)
        },
        "top_3": top3,
        "status": "success"
    })