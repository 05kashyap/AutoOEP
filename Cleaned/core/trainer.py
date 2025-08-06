"""
Training utilities for temporal models
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve


class ModelTrainer:
    """Handles training of temporal models"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train model for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted = (torch.sigmoid(output.squeeze()) > 0.5).float()
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader, criterion):
        """Validate model for one epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output.squeeze(), target)
                
                total_loss += loss.item()
                predicted = (torch.sigmoid(output.squeeze()) > 0.5).float()
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001, patience=10):
        """
        Train the model with early stopping
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            lr: Learning rate
            patience: Early stopping patience
            
        Returns:
            Training history dictionary
        """
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        print(f"Starting training on {self.device}")
        print("Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Time")
        print("-" * 60)
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            
            epoch_time = time.time() - epoch_start
            
            print(f"{epoch+1:5d} | {train_loss:10.4f} | {train_acc:9.2f} | "
                  f"{val_loss:8.4f} | {val_acc:7.2f} | {epoch_time:5.1f}s")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.1f} seconds")
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth', weights_only=False))
        
        return self.training_history
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                predictions = torch.sigmoid(output.squeeze())
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        auc_score = roc_auc_score(all_targets, all_predictions)
        
        # Binary predictions
        binary_preds = (all_predictions > 0.5).astype(int)
        accuracy = (binary_preds == all_targets).mean()
        
        return {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.training_history['train_loss'], label='Train Loss')
        ax1.plot(self.training_history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy plot
        ax2.plot(self.training_history['train_acc'], label='Train Accuracy')
        ax2.plot(self.training_history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, test_results):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(test_results['targets'], test_results['predictions'])
        auc_score = test_results['auc_score']
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.show()


class ModelSaver:
    """Handles saving and loading of trained models"""
    
    @staticmethod
    def save_model(model, scaler, model_path, scaler_path=None):
        """Save model and scaler"""
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_size': model.input_size if hasattr(model, 'input_size') else None,
                'hidden_size': model.hidden_size if hasattr(model, 'hidden_size') else None,
                'num_layers': model.num_layers if hasattr(model, 'num_layers') else None,
            },
            'scaler_mean': scaler.mean_,
            'scaler_scale': scaler.scale_
        }, model_path)
        
        print(f"Model saved to {model_path}")
        
        # Save scaler separately if path provided
        if scaler_path:
            import joblib
            joblib.dump(scaler, scaler_path)
            print(f"Scaler saved to {scaler_path}")
    
    @staticmethod
    def load_model(model_class, model_path, device='cpu'):
        """Load model and scaler"""
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Create model with saved config
        config = checkpoint['model_config']
        model = model_class(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Create and initialize scaler
        from core.data_handler import CustomScaler
        scaler = CustomScaler()
        scaler.mean_ = checkpoint['scaler_mean']
        scaler.scale_ = checkpoint['scaler_scale']
        
        return model, scaler
