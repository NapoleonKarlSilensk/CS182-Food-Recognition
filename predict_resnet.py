"""
predict test_a with ResNet model optimized
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from pathlib import Path
import librosa

# Configuration
MODEL_PATH = 'experiments/resnet_optimized_full/models/best_model.keras'
TEST_DIR = 'test_a'
OUTPUT_FILE = 'submissions/submission_resnet_test_a.csv'

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

def extract_features(audio_path, sr=SAMPLE_RATE, duration=DURATION):
    """Extract Mel spectrogram features"""
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
        expected_frames = 1 + (int(sr * duration) - N_FFT) // HOP_LENGTH
        return np.zeros((N_MELS, expected_frames))

def main():
    print("=" * 80)
    print("ResNet - Test_a Prediction")
    print("=" * 80)
    
    # Load test data
    print(f"\n[Step 1/4] Loading test data from {TEST_DIR}...")
    test_path = Path(TEST_DIR)
    audio_files = sorted(test_path.glob("*.wav"))
    print(f"[OK] Found {len(audio_files)} audio files")
    
    # Extract features
    print("\n[Step 2/4] Extracting features...")
    features = []
    file_names = []
    
    for i, audio_file in enumerate(audio_files):
        if (i + 1) % 100 == 0:
            print(f"  Processing: {i + 1}/{len(audio_files)}")
        
        feature = extract_features(str(audio_file))
        feature = np.expand_dims(feature, axis=-1)  # Add channel dimension
        features.append(feature)
        file_names.append(audio_file.name)
    
    X_test = np.array(features)
    print(f"[OK] Test data shape: {X_test.shape}")
    
    # Load model
    print(f"\n[Step 3/4] Loading model...")
    print(f"  Model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"[OK] Model loaded successfully")
    
    # Make predictions
    print("\n[Step 4/4] Making predictions...")
    predictions = model.predict(X_test, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    
    print(f"\n[OK] Predictions completed!")
    print(f"  Average confidence: {np.mean(confidences):.4f}")
    print(f"  Min confidence: {np.min(confidences):.4f}")
    print(f"  Max confidence: {np.max(confidences):.4f}")
    
    # Save results
    predicted_labels = [CLASS_LABELS[pred] for pred in predicted_classes]
    df = pd.DataFrame({
        'name': file_names,
        'label': predicted_labels
    })
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[OK] Predictions saved to: {OUTPUT_FILE}")
    
    # Print prediction distribution
    print("\nPrediction distribution:")
    from collections import Counter
    label_counts = Counter(predicted_labels)
    for label in CLASS_LABELS:
        count = label_counts[label]
        percentage = (count / len(predicted_labels)) * 100
        print(f"  {label:15s}: {count:4d} ({percentage:5.2f}%)")

if __name__ == "__main__":
    main()
