"""
predict test_a with EfficientNet model optimized
"""

import tensorflow as tf
import numpy as np
import os
import librosa
from pathlib import Path
import pandas as pd
from datetime import datetime

# GPU settings
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU enabled: {len(gpus)} GPU(s) available")
    except RuntimeError as e:
        print(f"GPU error: {e}")

# Experiment configuration
EXP_NAME = "efficientnet_full"
MODEL_PATH = f"experiments/{EXP_NAME}/models/best_model.keras"

# Data configuration - consistent with training
SAMPLE_RATE = 16000
DURATION = 5
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# Class mapping
CLASSES = sorted([
    'aloe', 'burger', 'cabbage', 'candied_fruits', 'carrots',
    'chips', 'chocolate', 'drinks', 'fries', 'grapes',
    'gummies', 'ice-cream', 'jelly', 'noodles', 'pickles',
    'pizza', 'ribs', 'salmon', 'soup', 'wings'
])

def extract_mel_spectrogram(audio_path, sr=SAMPLE_RATE, duration=DURATION):
    """Extract Mel spectrogram features"""
    try:
        # Load audio
        y, sr_loaded = librosa.load(audio_path, sr=sr, duration=duration)
        
        # Ensure fixed length
        target_length = sr * duration
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            y = y[:target_length]
        
        # Extract Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        
        # Convert to decibels
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        return mel_spec_norm
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        # Return silent frames
        expected_frames = 1 + (sr * duration - N_FFT) // HOP_LENGTH
        return np.zeros((N_MELS, expected_frames))

def load_test_data(test_dir):
    """Load test data"""
    print(f"\n[Step 1/4] Loading test data from {test_dir}...")
    
    audio_files = []
    file_names = []
    
    test_path = Path(test_dir)
    if not test_path.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    # Collect all .wav files
    for audio_file in sorted(test_path.glob("*.wav")):
        audio_files.append(str(audio_file))
        file_names.append(audio_file.name)
    
    print(f"✓ Found {len(audio_files)} audio files")
    return audio_files, file_names

def preprocess_test_data(audio_files):
    """Preprocess test data"""
    print("\n[Step 2/4] Extracting features...")
    
    features = []
    for i, audio_path in enumerate(audio_files):
        if (i + 1) % 100 == 0:
            print(f"  Processing: {i + 1}/{len(audio_files)}")
        
        mel_spec = extract_mel_spectrogram(audio_path)
        features.append(mel_spec)
    
    # Convert to numpy array and add channel dimension
    X_test = np.array(features)
    X_test = np.expand_dims(X_test, axis=-1)  # (N, 128, 157, 1)
    
    print(f"✓ Test data shape: {X_test.shape}")
    return X_test

def main():
    print("=" * 80)
    print("EfficientNet - Test_a Prediction")
    print("=" * 80)
    
    # Check model file
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    # Load test data
    audio_files, file_names = load_test_data("test_a")
    
    # Preprocess
    X_test = preprocess_test_data(audio_files)
    
    # Load model
    print("\n[Step 3/4] Loading model...")
    print(f"  Model: {MODEL_PATH}")
    
    # Use compile=False to avoid .keras format issues
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✓ Model loaded successfully")
    
    # Predict
    print("\n[Step 4/4] Making predictions...")
    predictions = model.predict(X_test, batch_size=32, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_labels = [CLASSES[idx] for idx in predicted_classes]
    
    # Calculate confidence
    confidences = np.max(predictions, axis=1)
    avg_confidence = np.mean(confidences)
    
    print(f"\nPredictions completed!")
    print(f"  Average confidence: {avg_confidence:.4f}")
    print(f"  Min confidence: {np.min(confidences):.4f}")
    print(f"  Max confidence: {np.max(confidences):.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"submissions/submission_efficientnet_test_a_{timestamp}.csv"
    
    submission_df = pd.DataFrame({
        'name': file_names,
        'label': predicted_labels
    })
    submission_df.to_csv(output_file, index=False)
    
    print(f"\n{'=' * 80}")
    print(f"[OK] Predictions saved to: {output_file}")
    print(f"{'=' * 80}")
    
    # Display prediction distribution
    print("\nPrediction distribution:")
    from collections import Counter
    label_counts = Counter(predicted_labels)
    for label in CLASSES:
        count = label_counts.get(label, 0)
        percentage = count / len(predicted_labels) * 100
        print(f"  {label:15s}: {count:3d} ({percentage:5.2f}%)")
    
    print("\n" + "=" * 80)
    print("=" * 80)

if __name__ == "__main__":
    main()
