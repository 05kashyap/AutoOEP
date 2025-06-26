import torch
import torch.nn as nn
class LSTMModel(nn.Module):
    """LSTM-based sequence model for cheat detection"""
    
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        
        self.fc1 = nn.Linear(hidden_size2, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # First LSTM layer
        x, _ = self.lstm1(x)
        batch_size, seq_len, hidden_size = x.size()
        x = x.reshape(-1, hidden_size)
        x = self.dropout1(x)
        x = self.bn1(x)
        x = x.reshape(batch_size, seq_len, hidden_size)
        
        # Second LSTM layer
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # Take only the last output
        x = self.dropout2(x)
        x = self.bn2(x)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class GRUModel(nn.Module):
    """GRU-based sequence model for cheat detection"""
    
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, output_size=1):
        super(GRUModel, self).__init__()
        self.gru1 = nn.GRU(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        
        self.gru2 = nn.GRU(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        
        self.fc1 = nn.Linear(hidden_size2, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # First GRU layer
        x, _ = self.gru1(x)
        batch_size, seq_len, hidden_size = x.size()
        x = x.reshape(-1, hidden_size)
        x = self.dropout1(x)
        x = self.bn1(x)
        x = x.reshape(batch_size, seq_len, hidden_size)
        
        # Second GRU layer
        x, _ = self.gru2(x)
        x = x[:, -1, :]  # Take only the last output
        x = self.dropout2(x)
        x = self.bn2(x)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x