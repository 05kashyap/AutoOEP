import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    """LSTM-based sequence classifier with multiple pooling options."""
    
    def __init__(self, input_size, hidden_size=64, fc_hidden=32, 
                 output_size=1, dropout=0.35, pooling="last"):
        super(LSTMModel, self).__init__()
        
        self.pooling = pooling  # "last", "mean", or "attention"
        
        # LSTM encoder
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Attention mechanism (only used if pooling="attention")
        if pooling == "attention":
            self.attn = nn.Linear(hidden_size, 1)
        
        # Normalization + dropout
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Classifier
        self.fc1 = nn.Linear(hidden_size, fc_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden, output_size)
        

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)   # out: (batch, seq_len, hidden)
        
        # ---- Pooling options ----
        if self.pooling == "last":
            x = h_n[-1]  # last hidden state: (batch, hidden)
            
        elif self.pooling == "mean":
            x = out.mean(dim=1)  # mean over time: (batch, hidden)
            
        elif self.pooling == "attention":
            # Compute attention weights
            attn_weights = F.softmax(self.attn(out), dim=1)  # (batch, seq_len, 1)
            # Weighted sum of LSTM outputs
            x = torch.sum(out * attn_weights, dim=1)  # (batch, hidden)
            
        else:
            raise ValueError("Invalid pooling type. Choose from 'last', 'mean', 'attention'.")
        
        # ---- Classifier ----
        x = self.layernorm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)   # logits (batch, 1)
        
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
        # self.sigmoid = nn.Sigmoid()  # Removed for consistency with LSTM
        
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
        # x = self.sigmoid(x)  # Removed
        return x