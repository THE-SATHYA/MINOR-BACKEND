from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
import librosa
import numpy as np
import io
from model_loader import load_model, preprocess_audio
from fastapi.responses import FileResponse
import os

# Initialize FastAPI app
app = FastAPI()

# Load trained model
model = load_model("models/best_model.pth")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded audio file
        audio_data = await file.read()
        
        # Preprocess audio
        mfcc_tensor = extract_mfcc(audio_data)
        
        # Make prediction
        with torch.no_grad():
            output = model(mfcc_tensor)
            pred_idx = output.argmax(dim=1).item()
            label_map = {0: "flow", 1: "breathy", 2: "neutral", 3: "pressed"}
            predicted_label = label_map.get(pred_idx, "unknown")
        
        return {"filename": file.filename, "predicted_class": predicted_label}
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
