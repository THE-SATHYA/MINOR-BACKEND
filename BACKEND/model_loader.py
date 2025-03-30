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

def preprocess_audio(audio_data):
    """
    Convert uploaded audio bytes to MFCC tensor for model prediction.
    """
    # Load audio from bytes
    y, sr = librosa.load(io.BytesIO(audio_data), sr=16000, mono=True)
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs = np.expand_dims(mfccs, axis=0)  # Add batch dimension

    # Convert to torch tensor
    return torch.tensor(mfccs, dtype=torch.float32)

# Preprocess audio file
def predict_audio(file_data, sr=16000, n_mfcc=13, padding_length=100):
    audio_file = io.BytesIO(file_data)
    y, _ = librosa.load(audio_file, sr=sr, mono=True)
    if y.size == 0:
        raise ValueError("Loaded audio is empty")
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-6)
    
    if mfcc.shape[1] < padding_length:
        mfcc = np.pad(mfcc, ((0, 0), (0, padding_length - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :padding_length]
    
    mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return mfcc_tensor
