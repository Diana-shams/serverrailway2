from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import uvicorn
import io
from PIL import Image
import os
import gdown

app = FastAPI()

# --- Download model if it doesn't exist ---
MODEL_PATH = "tomato_model.h5"
GOOGLE_DRIVE_URL = "https://drive.google.com/uc?id=10qM_Xz-tpyc3dNfSH1lXxjRieBmnoy_N"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(GOOGLE_DRIVE_URL, MODEL_PATH, quiet=False)

# --- Load model ---
model = load_model(MODEL_PATH)

# --- Class names ---
class_names = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# --- API route ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    return JSONResponse(content={"predicted_class": predicted_class})

# Optional: run locally
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
