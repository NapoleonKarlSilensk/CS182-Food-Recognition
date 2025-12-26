# -*- coding: utf-8 -*-
"""
predict test_a with CNN model optimized
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm

from config import *
from data_preprocessing import ADL


print("="*80)
print("CNN Optimized - Predict test_a")
print("="*80)

# Model path
model_path = "experiments/cnn_optimized_full/models/best_model.keras"
test_dir = "test_a"

print(f"\n[1/4] Loading model...")
print(f"  Model path: {model_path}")

# Load using native TensorFlow
model = tf.keras.models.load_model(model_path, compile=False)
print("[OK] Model loaded successfully")
print(f"  Validation accuracy: 89.71%")

print(f"\n[2/4] Loading test data...")
print(f"  Test directory: {test_dir}")
loader = ADL(test_dir, sample_rate=SAMPLE_RATE, duration=DURATION)
X_test, filenames = loader.load_test(test_dir)

print(f"[OK] Test data loaded successfully")
print(f"  Number of samples: {len(X_test)}")
print(f"  Data shape: {X_test.shape}")

print(f"\n[3/4] Starting prediction...")
predictions = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
pred_classes = np.argmax(predictions, axis=1)

print("[OK] Prediction completed")
print(f"\n[4/4] Generating submission file...")
# Create results DataFrame
results = []
for filename, pred_idx in zip(filenames, pred_classes):
    results.append({
        'name': filename,
        'label': CLASS_NAMES[pred_idx]
    })

df = pd.DataFrame(results)

# Save to submissions directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"submissions/submission_cnn_test_a_{timestamp}.csv"
os.makedirs("submissions", exist_ok=True)
df.to_csv(output_file, index=False, encoding='utf-8')

print(f"[OK] Submission file saved: {output_file}")

# Statistics of prediction distribution
print("\nPrediction class distribution:")
class_counts = df['label'].value_counts()
for class_name, count in class_counts.items():
    percentage = count / len(df) * 100
    print(f"  {class_name}: {count} ({percentage:.2f}%)")

print("\n" + "="*80)
print("Prediction completed!")
print("="*80)
print(f"\nSubmission file: {output_file}")