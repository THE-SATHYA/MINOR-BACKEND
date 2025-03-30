from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
import librosa
import numpy as np
import io

# Define the model architecture
class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.lstm = nn.LSTM(64, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, 4)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

# Load trained model
def load_model(model_path="models/best_model.pth"):
    device = torch.device("cpu")
    model = AudioClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Preprocess audio file
def extract_mfcc(audio_data, sr=16000, n_mfcc=13, padding_length=100):
    try:
        # Load the audio file
        audio_file = io.BytesIO(audio_data)
        y, _ = librosa.load(audio_file, sr=sr, mono=True)
        if y.size == 0:
            raise ValueError("Loaded audio is empty")
        
        # Compute MFCCs and normalize
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-6)
        
        # Pad or truncate
        if mfcc.shape[1] < padding_length:
            mfcc = np.pad(mfcc, ((0, 0), (0, padding_length - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :padding_length]
        
        # Convert to tensor
        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return mfcc_tensor
    except Exception as e:
        raise ValueError(f"Error in preprocessing: {e}")

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
