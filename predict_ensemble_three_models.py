"""
Three-Model CNN + ResNet + EfficientNet Ensemble Prediction Script
"""

import os
import numpy as np
import librosa
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import pandas as pd

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[OK] GPU enabled: {len(gpus)} GPU(s) available")
    except RuntimeError as e:
        print(f"[WARNING] GPU configuration error: {e}")
else:
    print("[INFO] No GPU found, using CPU")

# model configurations
MODELS = {
    'CNN': {
        'name': 'CNN Optimized',
        'path': 'experiments/cnn_optimized_full/models/best_model.keras',
        'weight': 0.10  # CNN weight
    },
    'ResNet': {
        'name': 'ResNet Optimized',
        'path': 'experiments/resnet_optimized_full/models/best_model.keras',
        'weight': 0.40  # ResNet weight
    },
    'EfficientNet': {
        'name': 'EfficientNet',
        'path': 'experiments/efficientnet_full/models/best_model.keras',
        'weight': 0.50  # EfficientNet weight
    }
}

# Class labels
CLASS_LABELS = [
    'aloe', 'burger', 'cabbage', 'candied_fruits', 'carrots',
    'chips', 'chocolate', 'drinks', 'fries', 'grapes',
    'gummies', 'ice-cream', 'jelly', 'noodles', 'pickles',
    'pizza', 'ribs', 'salmon', 'soup', 'wings'
]

# Feature extraction parameters
SAMPLE_RATE = 16000
DURATION = 5.0
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

def extract_features(audio_path, sr=SAMPLE_RATE, duration=DURATION, n_mels=N_MELS, 
                     hop_length=HOP_LENGTH, n_fft=N_FFT):
    """Extract Mel spectrogram features (consistent with training)"""
    try:
        # Load audio
        y, sr_loaded = librosa.load(audio_path, sr=sr, duration=duration)
        
        # Ensure fixed length
        target_length = int(sr * duration)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            y = y[:target_length]
        
        # Extract Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Convert to decibels
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1] - critical step!
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        return mel_spec_norm
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        # Return silent frames
        expected_frames = 1 + (int(sr * duration) - n_fft) // hop_length
        return np.zeros((n_mels, expected_frames))

def load_test_data(test_dir):
    """Load test data"""
    print(f"\n[Step 1/5] Loading test data from {test_dir}...")
    
    audio_files = []
    file_names = []
    
    test_path = Path(test_dir)
    if not test_path.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    for audio_file in sorted(test_path.glob("*.wav")):
        audio_files.append(str(audio_file))
        file_names.append(audio_file.name)
    
    print(f"[OK] Found {len(audio_files)} audio files")
    return audio_files, file_names

def preprocess_test_data(audio_files):
    """Preprocess test data"""
    print("\n[Step 2/5] Extracting features...")
    
    features = []
    for i, audio_path in enumerate(audio_files):
        if (i + 1) % 100 == 0:
            print(f"  Processing: {i + 1}/{len(audio_files)}")
        
        feature = extract_features(audio_path)
        # Add channel dimension
        feature = np.expand_dims(feature, axis=-1)
        features.append(feature)
    
    X_test = np.array(features)
    print(f"[OK] Test data shape: {X_test.shape}")
    
    return X_test

def load_models():
    """Load all models"""
    print("\n[Step 3/5] Loading models...")
    
    models = {}
    for name, config in MODELS.items():
        print(f"  Loading {config['name']}...")
        if not os.path.exists(config['path']):
            print(f"    [WARNING] Model not found: {config['path']}")
            continue
        
        try:
            model = tf.keras.models.load_model(config['path'], compile=False)
            models[name] = model
            print(f"    [OK] {config['name']} loaded (weight: {config['weight']})")
        except Exception as e:
            print(f"    [ERROR] Failed to load {config['name']}: {e}")
    
    print(f"[OK] Successfully loaded {len(models)}/{len(MODELS)} models")
    return models

def ensemble_predict(models, X_test):
    """Ensemble prediction"""
    print(f"\n[Step 4/5] Making ensemble predictions...")
    
    all_probs = []
    weights = []
    
    for name, model in models.items():
        print(f"  Predicting with {MODELS[name]['name']}...")
        probs = model.predict(X_test, verbose=1)
        all_probs.append(probs)
        weights.append(MODELS[name]['weight'])
        print(f"    [OK] Done (avg confidence: {np.mean(np.max(probs, axis=1)):.4f})")
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    print(f"\n  Normalized weights: {dict(zip(models.keys(), weights))}")
    
    # Weighted average
    ensemble_probs = np.zeros_like(all_probs[0])
    for i, probs in enumerate(all_probs):
        ensemble_probs += weights[i] * probs
    
    # Get final predictions
    predictions = np.argmax(ensemble_probs, axis=1)
    confidences = np.max(ensemble_probs, axis=1)
    
    return predictions, confidences, ensemble_probs

def save_predictions(file_names, predictions, confidences, output_file):
    """Save predictions"""
    print(f"\n[Step 5/5] Saving predictions...")
    
    predicted_labels = [CLASS_LABELS[pred] for pred in predictions]
    
    df = pd.DataFrame({
        'name': file_names,
        'label': predicted_labels
    })
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"[OK] Predictions saved to: {output_file}")
    return predicted_labels

def print_statistics(predicted_labels, confidences):
    """Print statistics"""
    print("\n" + "=" * 80)
    print("[OK] Three-Model Ensemble predictions completed!")
    print("=" * 80)
    
    print(f"\nConfidence statistics:")
    print(f"  Average confidence: {np.mean(confidences):.4f}")
    print(f"  Min confidence: {np.min(confidences):.4f}")
    print(f"  Max confidence: {np.max(confidences):.4f}")
    
    print(f"\nPrediction distribution:")
    from collections import Counter
    label_counts = Counter(predicted_labels)
    for label in CLASS_LABELS:
        count = label_counts[label]
        percentage = (count / len(predicted_labels)) * 100
        print(f"  {label:15s}: {count:4d} ({percentage:5.2f}%)")

def main():
    print("\nModels:")
    for name, config in MODELS.items():
        print(f"  {config['name']}: weight = {config['weight']}")
    
    # Load test data
    audio_files, file_names = load_test_data("test_a")
    
    # Preprocess
    X_test = preprocess_test_data(audio_files)
    
    # Load models
    models = load_models()
    
    if len(models) == 0:
        print("\n[ERROR] No models loaded successfully!")
        return
    
    # Ensemble prediction
    predictions, confidences, ensemble_probs = ensemble_predict(models, X_test)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"submissions/submission_ensemble_three_models_test_a_{timestamp}.csv"
    predicted_labels = save_predictions(file_names, predictions, confidences, output_file)
    
    # Print statistics
    print_statistics(predicted_labels, confidences)

if __name__ == "__main__":
    main()
