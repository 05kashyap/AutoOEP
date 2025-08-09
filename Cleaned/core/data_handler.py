"""
Data handling utilities for temporal analysis
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DataHandler:
    """Handles data loading and processing for video proctoring system"""
    
    def __init__(self):
        """Initialize data handler"""
        pass
    
    def load_training_data(self, training_data_dir):
        """Load training data from directory (placeholder implementation)"""
        # This is a placeholder - in a real implementation, this would load actual training data
        # For now, return empty data to avoid import errors
        return [], []


class CustomScaler:
    """Simple scaler that standardizes features"""
    
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        
    def fit(self, X):
        """Fit the scaler to data"""
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self
        
    def transform(self, X):
        """Transform data using fitted scaler"""
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler not fitted. Call fit before using transform.")
        return (X - self.mean_) / self.scale_
        
    def fit_transform(self, X):
        """Fit scaler and transform data in one step"""
        return self.fit(X).transform(X)
        
    def initialize(self, mean, scale):
        """Initialize the scaler with provided mean and scale values"""
        self.mean_ = np.array(mean)
        self.scale_ = np.array(scale)
        return self


class TemporalDataset(Dataset):
    """Dataset class for temporal sequence data"""
    
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class DataProcessor:
    """Handles data loading and preprocessing for temporal analysis"""
    
    @staticmethod
    def custom_train_test_split(X, y, test_size=0.2, random_state=None):
        """Custom implementation of train_test_split"""
        if random_state is not None:
            np.random.seed(random_state)
            
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        
        # Generate random indices
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        return (X[train_indices], X[test_indices], 
                y[train_indices], y[test_indices])
    
    @staticmethod
    def create_sequences(data, window_size):
        """
        Create sequences from temporal data
        
        Args:
            data: Input data array
            window_size: Size of temporal window
            
        Returns:
            Array of sequences
        """
        sequences = []
        for i in range(len(data) - window_size + 1):
            sequences.append(data[i:i + window_size])
        return np.array(sequences)
    
    @staticmethod
    def load_and_preprocess_data(csv_path, window_size, scaler=None):
        """
        Load data from CSV and create sequences
        
        Args:
            csv_path: Path to CSV file
            window_size: Size of temporal window
            scaler: Optional scaler to use
            
        Returns:
            Tuple of (sequences, labels, scaler)
        """
        # Load data
        df = pd.read_csv(csv_path)
        
        # Separate features and labels
        if 'is_cheating' in df.columns:
            features = df.drop(['is_cheating'], axis=1)
            labels = df['is_cheating'].values
        else:
            raise ValueError("'is_cheating' column not found in data")
        
        # Scale features
        if scaler is None:
            scaler = CustomScaler()
            features_scaled = scaler.fit_transform(features.values)
        else:
            features_scaled = scaler.transform(features.values)
        
        # Create sequences
        sequences = DataProcessor.create_sequences(features_scaled, window_size)
        sequence_labels = labels[window_size - 1:]  # Labels for the last frame of each sequence
        
        return sequences, sequence_labels, scaler
    
    @staticmethod
    def create_data_loaders(X_train, X_test, y_train, y_test, batch_size=32):
        """Create PyTorch data loaders"""
        train_dataset = TemporalDataset(X_train, y_train)
        test_dataset = TemporalDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
