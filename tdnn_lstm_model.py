import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

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
        
        # TDNN blocks
        self.tdnn1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )
        self.tdnn2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )
        self.tdnn3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )

        # LSTM layer
        self.lstm = nn.LSTM(input_size=512, hidden_size=lstm_hidden_size,
                            num_layers=num_lstm_layers, batch_first=True, dropout=0.3)
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(lstm_hidden_size, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)

        x = x.transpose(1, 2)  # Shape for LSTM
        x, (hn, cn) = self.lstm(x)
        x = hn[-1]  # Last time step hidden state
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
