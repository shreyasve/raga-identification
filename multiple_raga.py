import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import torch.nn.functional as F
import joblib  # For saving and loading models, scalers, and selectors


# Dataset Class for Multi-Label Classification
class RagaDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)  # y as float32 for multi-label

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# TDNN + LSTM Model Class for Multi-Label Classification
class TDNN_LSTM(nn.Module):
    def __init__(self, input_dim, num_classes, lstm_hidden_size=256, num_lstm_layers=1):
        super(TDNN_LSTM, self).__init__()
        self.tdnn1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3)
        )
        self.tdnn2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3)
        )
        self.tdnn3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3)
        )
        self.lstm = nn.LSTM(512, lstm_hidden_size, num_layers=num_lstm_layers, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(lstm_hidden_size, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = x.transpose(1, 2)
        x, (hn, cn) = self.lstm(x)
        x = hn[-1]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x)  # Apply sigmoid for multi-label classification


# Load Data for Multi-Label Classification
def load_data_multi_label(file_path, n_mfcc=40):
    df = pd.read_csv(file_path)
    X = df.drop(columns=["filename", "raga"], errors="ignore")
    ragas = df["raga"].str.get_dummies(sep=",")  # Multi-label one-hot encoding
    y = ragas  # Use the one-hot-encoded matrix as `y`

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # OPTIONAL: Skip feature selection for now
    selector = None

    return X_scaled, y.values, scaler, selector


# Compute Class Weights (Optional for Multi-Label Classification)
def get_class_weights(y_encoded, num_classes):
    # Multi-label classification may not need this step, as we use sigmoid loss
    return torch.ones(num_classes, dtype=torch.float32)


# Train Model for Multi-Label Classification
def train_model_multi_label(model, train_loader, test_loader, num_epochs=50, lr=0.001, model_save_path="model.pth"):
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss for Multi-Label Classification
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(next(model.parameters()).device), labels.to(next(model.parameters()).device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total += labels.size(0)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / total:.4f}")
    torch.save(model.state_dict(), model_save_path)  # Save model to .pth file
    print("Model saved to", model_save_path)


# Predict Raga for Multi-Label Classification
def predict_multi_label_raga(model, csv_file, scaler):
    df = pd.read_csv(csv_file)
    X = df.drop(columns=["filename", "raga"], errors="ignore")
    X_scaled = scaler.transform(X)
    features = torch.tensor(X_scaled, dtype=torch.float32).to(next(model.parameters()).device)
    model.eval()
    with torch.no_grad():
        outputs = model(features)
        predictions = torch.sigmoid(outputs) > 0.5  # Threshold at 0.5 for multi-label classification
        predicted_labels = predictions.cpu().numpy()

    # Convert to list of ragas
    ragas = df["raga"].str.get_dummies(sep=",").columns
    predicted_ragas = []
    for label in predicted_labels:
        predicted_ragas.append([raga for i, raga in enumerate(ragas) if label[i]])

    return predicted_ragas


# Main Function for Training and Prediction
def main():
    parser = argparse.ArgumentParser(description="Raga Detection System")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "predict"], help="Mode: train or predict")
    parser.add_argument("--train_file", type=str, help="Path to training CSV file")
    parser.add_argument("--predict_file", type=str, help="Path to prediction CSV file")
    parser.add_argument("--model_save_path", type=str, default="model.pth", help="Path to save/load model")
    args = parser.parse_args()

    if args.mode == "train":
        if not args.train_file:
            raise ValueError("Training file path is required for training mode.")
        X, y, scaler, selector = load_data_multi_label(args.train_file)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        train_dataset = RagaDataset(X_train, y_train)
        test_dataset = RagaDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        num_classes = y.shape[1]  # Number of labels (ragas)
        model = TDNN_LSTM(input_dim=X_train.shape[1], num_classes=num_classes).to("cuda" if torch.cuda.is_available() else "cpu")
        train_model_multi_label(model, train_loader, test_loader, num_epochs=50, lr=0.001, model_save_path=args.model_save_path)
        joblib.dump(scaler, "scaler.pkl")  # Save the scaler for later use
        print("Preprocessing objects saved to scaler.pkl")

    elif args.mode == "predict":
        if not args.predict_file:
            raise ValueError("Prediction file path is required for prediction mode.")
        model = TDNN_LSTM(input_dim=24, num_classes=len(pd.read_csv(args.predict_file)["raga"].str.get_dummies(sep=",").columns))
        model.load_state_dict(torch.load(args.model_save_path))
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        scaler = joblib.load("scaler.pkl")
        predicted_ragas = predict_multi_label_raga(model, args.predict_file, scaler)
        print(f"Predicted Ragas for each part: {predicted_ragas}")


if __name__ == "__main__":
    main()
