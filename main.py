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
from tensorflow.keras import models, layers, regularizers, optimizers
from lime import lime_tabular


# Dataset Class
class RagaDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# TDNN + LSTM Model Class
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
        return x


# Load Data
def load_data(file_path, n_mfcc=40):
    df = pd.read_csv(file_path)
    X = df.drop(columns=['filename', 'raga'])
    y = df['raga'].values
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    selector = SelectKBest(f_classif, k=24)
    X_selected = selector.fit_transform(X_scaled, y_encoded)
    return X_selected, y_encoded, le, scaler, selector


# Compute Class Weights
def get_class_weights(y_encoded, num_classes):
    class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=y_encoded)
    return torch.tensor(class_weights, dtype=torch.float32)


# Train Model
def train_model(model, train_loader, test_loader, y_train, num_epochs=50, lr=0.001, model_save_path="model.pth"):
    criterion = nn.CrossEntropyLoss(weight=get_class_weights(y_train, len(np.unique(y_train)))).to(next(model.parameters()).device)
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
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {correct / total * 100:.2f}%")
    torch.save(model.state_dict(), model_save_path)  # Save model to .pth file
    print("Model saved to", model_save_path)


# Predict Raga
def predict_raga(model, csv_file, label_encoder, scaler, selector):
    df = pd.read_csv(csv_file)
    X = df.drop(columns=['filename', 'raga'], errors='ignore')
    X_scaled = scaler.transform(X)
    X_selected = selector.transform(X_scaled)
    features = torch.tensor(X_selected, dtype=torch.float32).to(next(model.parameters()).device)
    model.eval()
    with torch.no_grad():
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        predictions = label_encoder.inverse_transform(predicted.cpu().numpy())
    final_raga = max(set(predictions), key=predictions.tolist().count)
    return final_raga, predictions


# Keras Model for LIME Explanation
def create_keras_model(X_train, y_train, X_val, y_val):
    model1a = models.Sequential()
    model1a.add(layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
    model1a.add(layers.BatchNormalization())
    model1a.add(layers.Dropout(0.5))
    model1a.add(layers.Dense(256, activation='relu'))
    model1a.add(layers.BatchNormalization())
    model1a.add(layers.Dropout(0.5))
    model1a.add(layers.Dense(128, activation='relu'))
    model1a.add(layers.BatchNormalization())
    model1a.add(layers.Dropout(0.5))
    model1a.add(layers.Dense(64, activation='relu'))
    model1a.add(layers.BatchNormalization())
    model1a.add(layers.Dropout(0.5))
    model1a.add(layers.Dense(32, activation='softmax'))

    model1a.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    model1a.fit(X_train, y_train, batch_size=128, epochs=200, validation_data=(X_val, y_val))
    return model1a


def lime_explanation(X_train, X_test, model1a):
    explainer = lime_tabular.LimeTabularExplainer(X_train, feature_names=X_train.columns, class_names=['target'])
    def predict_fn(x):
        return model1a.predict(x)
    
    explanation = explainer.explain_instance(X_test[0], predict_fn, num_features=23)
    print("Explanation:")
    print(explanation)

    # Print the feature importances
    print("Feature Importances:")
    feature_importances = explanation.as_list()
    for feature, importance in feature_importances:
        print(f"{feature}: {importance}")


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
        X, y_encoded, le, scaler, selector = load_data(args.train_file)
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        train_dataset = RagaDataset(X_train, y_train)
        test_dataset = RagaDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        num_classes = len(np.unique(y_encoded))
        model = TDNN_LSTM(input_dim=X_train.shape[1], num_classes=num_classes).to("cuda" if torch.cuda.is_available() else "cpu")
        train_model(model, train_loader, test_loader, y_train, num_epochs=100, lr=0.001, model_save_path=args.model_save_path)
        joblib.dump((le, scaler, selector), "preprocessing.pkl")
        print("Preprocessing objects saved to preprocessing.pkl")

    elif args.mode == "predict":
        if not args.predict_file:
            raise ValueError("Prediction file path is required for prediction mode.")
        
        # Load preprocessing objects
        le, scaler, selector = joblib.load("preprocessing.pkl")
        
        # Fix: Define y_encoded using the label encoder
        num_classes = len(le.classes_)  # Get number of classes from label encoder
        
        model = TDNN_LSTM(input_dim=24, num_classes=num_classes).to("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(args.model_save_path))
        model.eval()

        final_raga, predictions = predict_raga(model, args.predict_file, le, scaler, selector)
        print(f"The final predicted raga is: {final_raga}")
        print(f"Predicted Ragas for each part: {predictions}")
if __name__ == "__main__":
    main()
