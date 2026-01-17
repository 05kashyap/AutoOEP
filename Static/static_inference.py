import os
import joblib
import numpy as np
import pandas as pd


def load_model_and_metadata(model_path):
	"""
	Load a saved model, scaler, and metadata
	"""
	# Load the model
	model = joblib.load(model_path)

	# Try to load metadata and scaler robustly using timestamp at end of filename
	try:
		import re as _re
		base = os.path.basename(model_path)
		dirn = os.path.dirname(model_path)
		m = _re.search(r"_(\d{8}_\d{6})\.pkl$", base)
		if not m:
			raise ValueError("Could not extract timestamp from model filename")
		ts = m.group(1)
		metadata_path = os.path.join(dirn, f"model_metadata_{ts}.pkl")
		metadata = joblib.load(metadata_path)

		# Load scaler if it was used
		scaler = None
		if metadata.get('scaler_used', False):
			scaler_path = metadata.get('scaler_path')
			if scaler_path and os.path.exists(scaler_path):
				scaler = joblib.load(scaler_path)
				print(f"Scaler loaded successfully")
			else:
				fallback = os.path.join(dirn, f"scaler_{ts}.pkl")
				if os.path.exists(fallback):
					scaler = joblib.load(fallback)
					print(f"Scaler loaded from fallback path")
				else:
					print("Warning: Scaler was expected but not found")

		print(f"Model and metadata loaded successfully")
		print(f"Model trained on: {metadata['training_timestamp']}")
		print(f"Feature count: {metadata['feature_count']}")
		print(f"Scaler used: {metadata.get('scaler_used', False)}")

		return model, metadata, scaler

	except Exception as e:
		print(f"Model loaded successfully, but metadata/scaler loading failed: {str(e)}")
		return model, None, None


def predict_cheating(model, sample_data, feature_names=None, metadata=None, scaler=None):
	"""
	Make predictions on new sample data with proper preprocessing

	Parameters:
	- model: trained XGBoost model
	- sample_data: pandas DataFrame or numpy array with features
	- feature_names: list of feature names (optional, for validation)
	- metadata: model metadata (optional)
	- scaler: fitted scaler object (optional)

	Returns:
	- predictions: binary predictions (0/1)
	- probabilities: probability of cheating
	"""
	# Convert to DataFrame if it's not already
	if not isinstance(sample_data, pd.DataFrame):
		if feature_names is not None:
			sample_data = pd.DataFrame(sample_data, columns=feature_names)
		else:
			sample_data = pd.DataFrame(sample_data)

	# Validate features if metadata is available
	if metadata is not None:
		expected_features = metadata['feature_names']
		current_features = list(sample_data.columns)

		# Check if all expected features are present
		missing_features = set(expected_features) - set(current_features)
		if missing_features:
			print(f"Warning: Missing features: {missing_features}")

		# Reorder columns to match training data
		sample_data = sample_data.reindex(columns=expected_features, fill_value=0)

	# Handle missing values (use same approach as training)
	sample_data_processed = sample_data.fillna(sample_data.median())

	# Apply scaling if scaler is provided
	if scaler is not None:
		sample_data_processed = pd.DataFrame(
			scaler.transform(sample_data_processed),
			columns=sample_data_processed.columns,
			index=sample_data_processed.index
		)
		print("Applied scaling to input data")

	# Make predictions
	predictions = model.predict(sample_data_processed)
	probabilities = model.predict_proba(sample_data_processed)[:, 1]

	return predictions, probabilities


def demonstrate_prediction(model_path, sample_csv_path=None):
	"""
	Demonstrate how to use the saved model for predictions
	"""
	print("\n" + "="*50)
	print("PREDICTION DEMONSTRATION")
	print("="*50)

	# Load the model, metadata, and scaler
	model, metadata, scaler = load_model_and_metadata(model_path)

	if sample_csv_path and os.path.exists(sample_csv_path):
		# Load sample data from CSV
		sample_data = pd.read_csv(sample_csv_path)

		# Remove timestamp and target if they exist
		columns_to_remove = ['timestamp', 'is_cheating']
		for col in columns_to_remove:
			if col in sample_data.columns:
				sample_data = sample_data.drop(columns=[col])

		print(f"Making predictions on {len(sample_data)} samples from {sample_csv_path}")

	else:
		# Create a sample data point for demonstration
		if metadata:
			feature_names = metadata['feature_names']
		else:
			# Default feature names if metadata not available
			feature_names = [
				'verification_result', 'num_faces', 'iris_pos', 'iris_ratio',
				'mouth_zone', 'mouth_area', 'x_rotation', 'y_rotation', 'z_rotation',
				'radial_distance', 'gaze_direction', 'gaze_zone', 'watch', 'headphone',
				'closedbook', 'earpiece', 'cell phone', 'openbook', 'chits', 'sheet',
				'H-Distance', 'F-Distance'
			]

		# Create sample data
		sample_data = pd.DataFrame({
			'verification_result': [1],
			'num_faces': [1],
			'iris_pos': [0],
			'iris_ratio': [0.5],
			'mouth_zone': [0],
			'mouth_area': [25.0],
			'x_rotation': [0],
			'y_rotation': [0],
			'z_rotation': [0],
			'radial_distance': [5000.0],
			'gaze_direction': [2],
			'gaze_zone': [2],
			'watch': [0],
			'headphone': [0],
			'closedbook': [0],
			'earpiece': [0],
			'cell phone': [1],
			'openbook': [0],
			'chits': [0],
			'sheet': [0],
			'H-Distance': [100.0],
			'F-Distance': [1000.0]
		})

		print("Making predictions on sample data:")
		print(sample_data)

	# Make predictions with proper preprocessing
	predictions, probabilities = predict_cheating(
		model, sample_data, metadata=metadata, scaler=scaler
	)

	# Display results
	print(f"\nPrediction Results:")
	print("-" * 30)
	for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
		status = "CHEATING" if pred == 1 else "NOT CHEATING"
		confidence = prob if pred == 1 else (1 - prob)
		print(f"Sample {i+1}: {status} (Confidence: {confidence:.3f})")
		print(f"  Cheating Probability: {prob:.3f}")

		# Risk assessment
		if prob > 0.8:
			risk = "HIGH RISK"
		elif prob > 0.5:
			risk = "MEDIUM RISK"
		elif prob > 0.3:
			risk = "LOW RISK"
		else:
			risk = "MINIMAL RISK"
		print(f"  Risk Level: {risk}")
		print()

	return predictions, probabilities

