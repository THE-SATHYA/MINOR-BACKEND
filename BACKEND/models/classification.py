import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import numpy as np
import re
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset

# Constants
MAX_FRAMES = 120  
BATCH_SIZE = 16  
LEARNING_RATE = 0.0003  
WEIGHT_DECAY = 1e-6  
DROPOUT_RATE = 0.1
EPOCHS = 150
K_FOLDS = 10  

# Load dataset
class AudioDataset(Dataset):
    def __init__(self, feature_file):
        with open(feature_file, "rb") as f:
            self.data = pickle.load(f)
        
        self.files = list(self.data.keys())
        self.mfccs = [self._pad_or_truncate(self.data[f]["mfccs"]) for f in self.files]
        self.labels = np.array([self._extract_label(f) for f in self.files])
    
    def _pad_or_truncate(self, mfcc):
        if mfcc.shape[1] > MAX_FRAMES:
            return mfcc[:, :MAX_FRAMES]
        else:
            pad_width = MAX_FRAMES - mfcc.shape[1]
            return np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    
    def _extract_label(self, filename):
        numbers = re.findall(r'\d+', filename)
        return int(numbers[-1]) % 4 if numbers else 0
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        mfcc = self.mfccs[idx]
        mfcc = np.expand_dims(mfcc, axis=0)  # Shape (1, 13, 100) for CNN
        return torch.tensor(mfcc, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# Attention Layer
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn_weights = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        attn_scores = self.attn_weights(lstm_output).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)
        attn_output = torch.sum(lstm_output * attn_weights.unsqueeze(-1), dim=1)
        return attn_output, attn_weights

# Model Definition
class AudioClassifier(nn.Module):
    def __init__(self, input_dim=13, num_classes=4):
        super(AudioClassifier, self).__init__()
        
        # CNN Feature Extraction
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d((2, 2))
        self.dropout_cnn = nn.Dropout(DROPOUT_RATE)
        
        # BiLSTM
        self.lstm = nn.LSTM(64 * (MAX_FRAMES // 4), 128, batch_first=True, bidirectional=True)
        self.dropout_lstm = nn.Dropout(DROPOUT_RATE)
        
        # Attention
        self.attention = Attention(256)
        
        # Fully Connected Layer
        self.fc = nn.Linear(256, num_classes)
        self.dropout_fc = nn.Dropout(DROPOUT_RATE)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout_cnn(x)
        
        x = x.permute(0, 2, 1, 3).reshape(x.size(0), x.size(2), -1)  # (Batch, Time, Features)
        
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout_lstm(lstm_out)
        attn_out, attn_weights = self.attention(lstm_out)
        attn_out = self.dropout_fc(attn_out)
        
        output = self.fc(attn_out)
        return output

# Train and Evaluate Model for Each Fold
def train_and_evaluate():
    dataset = AudioDataset("test.pkl")
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n===== Fold {fold + 1} =====")
        print(f"Validation Set: Fold {fold + 1}")
        print(f"Training Set: All other folds")

        # Create DataLoaders
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize Model
        model = AudioClassifier()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        # Training Loop
        for epoch in range(EPOCHS):
            model.train()
            total_loss, correct, total = 0, 0, 0

            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            train_accuracy = correct / total
            print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}, Train Accuracy = {train_accuracy:.2%}")

        # Validation
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_accuracy = correct / total
        fold_results.append(val_accuracy)
        print(f"Fold {fold + 1} Validation Accuracy: {val_accuracy:.2%}")

    # Print Final Results
    print("\n=== Final Cross-Validation Results ===")
    for i, acc in enumerate(fold_results):
        print(f"Fold {i + 1}: Accuracy = {acc:.2%}")
    print(f"Average Accuracy: {np.mean(fold_results):.2%}")

# Run Training and Evaluation
if __name__ == "__main__":
    train_and_evaluate()