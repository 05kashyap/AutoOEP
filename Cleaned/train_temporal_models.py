#!/usr/bin/env python3
"""
Train Temporal Models - Usage Example Script
Demonstrates how to train temporal models separately from inference
"""

import os
import sys

# Add project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "Proctor"))

from Proctor.temporal_trainer import TemporalModelTrainer

def train_lstm_model():
    """Example: Train LSTM model"""
    print("=== Training LSTM Model ===")
    
    # Configuration
    data_dir = "path/to/training/data"  # Update with actual path
    save_path = "models/lstm_temporal_model.pth"
    
    # Create trainer
    trainer = TemporalModelTrainer(
        model_type='LSTM',
        window_size=15,
        input_size=23
    )
    
    # Set training parameters
    trainer.num_epochs = 100
    trainer.batch_size = 32
    trainer.learning_rate = 0.001
    
    try:
        # Run complete training pipeline
        results = trainer.train_complete_pipeline(data_dir, save_path)
        
        print(f"\n✅ LSTM Training completed!")
        print(f"Final accuracy: {results['test_results']['test_accuracy']:.4f}")
        print(f"Model saved to: {save_path}")
        
        return save_path
        
    except Exception as e:
        print(f"❌ LSTM Training failed: {e}")
        return None

def train_gru_model():
    """Example: Train GRU model"""
    print("\n=== Training GRU Model ===")
    
    # Configuration
    data_dir = "path/to/training/data"  # Update with actual path
    save_path = "models/gru_temporal_model.pth"
    
    # Create trainer
    trainer = TemporalModelTrainer(
        model_type='GRU',
        window_size=15,
        input_size=23
    )
    
    # Set training parameters
    trainer.num_epochs = 100
    trainer.batch_size = 32
    trainer.learning_rate = 0.001
    
    try:
        # Run complete training pipeline
        results = trainer.train_complete_pipeline(data_dir, save_path)
        
        print(f"\n✅ GRU Training completed!")
        print(f"Final accuracy: {results['test_results']['test_accuracy']:.4f}")
        print(f"Model saved to: {save_path}")
        
        return save_path
        
    except Exception as e:
        print(f"❌ GRU Training failed: {e}")
        return None

def test_inference_with_trained_model(model_path):
    """Test inference using trained model"""
    print(f"\n=== Testing Inference with Trained Model ===")
    
    try:
        # Import the inference-only trainer
        from Proctor.temporal_proctor import TemporalProctor
        
        # Create inference trainer
        inference_trainer = TemporalProctor(window_size=15)
        
        # Load the trained model
        inference_trainer.load_models(model_path)
        
        # Test with sample features
        sample_features = {
            'H-Hand Detected': True,
            'F-Hand Detected': False,
            'H-Prohibited Item': False,
            'F-Prohibited Item': False,
            'verification_result': True,
            'num_faces': 1,
            'Cheat Score': 0.4,
            'H-Distance': 150.0,
            'F-Distance': 200.0
        }
        
        # Add features and get prediction
        inference_trainer.add_frame_features(sample_features)
        prediction = inference_trainer.get_temporal_prediction()
        stats = inference_trainer.get_statistics()
        
        print(f"✅ Inference test successful!")
        print(f"Prediction: {prediction:.4f}")
        print(f"Model type: {stats['model_type']}")
        print(f"Using PyTorch model: {stats['use_pytorch_model']}")
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")

def main():
    """Main execution"""
    print("Temporal Model Training and Testing Example")
    print("=" * 50)
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # NOTE: Update data paths before running
    print("\n⚠️  IMPORTANT: Update data_dir paths in the functions before running!")
    print("Current paths are placeholders and need to point to actual training data.")
    
    # Uncomment the sections you want to run:
    
    # 1. Train LSTM model
    # lstm_model_path = train_lstm_model()
    
    # 2. Train GRU model  
    # gru_model_path = train_gru_model()
    
    # 3. Test inference (use path from training or existing model)
    # if lstm_model_path:
    #     test_inference_with_trained_model(lstm_model_path)
    
    print("\n💡 Usage Instructions:")
    print("1. Update data_dir paths to point to your training data")
    print("2. Uncomment the training functions you want to run")
    print("3. Run this script to train models")
    print("4. Use trained models in VideoProctor for inference")
    
    print("\n📝 Command Line Usage:")
    print("python Proctor/temporal_trainer.py --data_dir <path> --model_type LSTM --save_path models/model.pth")

if __name__ == "__main__":
    main()
