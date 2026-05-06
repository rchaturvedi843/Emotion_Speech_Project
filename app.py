import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import librosa
import numpy as np
import joblib

app = FastAPI(title="Emotion Recognition from Speech")

# Setup paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
STATIC_DIR = os.path.join(PROJECT_DIR, "static")
TEMPLATES_DIR = os.path.join(PROJECT_DIR, "templates")

# Mount static and templates
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Load Models
MODEL_PATH = os.path.join(OUTPUT_DIR, "svm_model.pkl")
LE_PATH = os.path.join(OUTPUT_DIR, "label_encoder.pkl")

# Initialize models
try:
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LE_PATH)
    print("Model and LabelEncoder loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load models. Please train the model first. Error: {e}")
    model = None
    le = None

def extract_mfcc(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    if not model or not le:
        return JSONResponse(status_code=500, content={"error": "Model is not trained. Please train the model."})
    
    try:
        # Load audio from file bytes
        contents = await file.read()
        import io
        import soundfile as sf
        audio_io = io.BytesIO(contents)
        audio, sr = sf.read(audio_io)
        
        # If stereo, convert to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample to 22050 (librosa default) if needed, but sf.read just reads as is. 
        # librosa can handle feature extraction on different sample rates, but 
        # for consistency with training (librosa.load default is 22050):
        if sr != 22050:
            audio = librosa.resample(y=audio, orig_sr=sr, target_sr=22050)
            sr = 22050

        features = extract_mfcc(audio, sr)
        features = features.reshape(1, -1)
        
        # Predict
        pred_encoded = model.predict(features)
        pred_emotion = le.inverse_transform(pred_encoded)[0]
        
        return {"emotion": pred_emotion}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
