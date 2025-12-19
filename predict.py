# -*- coding: utf-8 -*-
"""
Prediction script: Use trained models to make predictions on test data
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
from datetime import datetime

from config import *
from data_preprocessing import ADL


def predict_and_save(model_path, test_dir, output_filename):
    """
    Use a trained model to make predictions and save the results
    
    Args:
        model_path: Path to the model file
        test_dir: Directory of test data
        output_filename: Output CSV file name
    """
    print("\n" + "="*80)
    print(f"Starting prediction")
    print("="*80)
    print(f"Model path: {model_path}")
    print(f"Test directory: {test_dir}")
    
    # 1. Load model
    print("\n[Step 1/4] Load model...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = keras.models.load_model(model_path)
    print("Model loaded successfully!")
    model.summary()
    
    # 2. Load test data
    print(f"\n[Step 2/4] Load test data...")
    loader = ADL(test_dir)
    X_test, file_names = loader.load_test_data(test_dir)
    
    print(f"Number of test samples: {len(file_names)}")
    
    # 3. Make predictions
    print(f"\n[Step 3/4] Make predictions...")
    predictions = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Convert to class names
    predicted_labels = [CLASS_NAMES[idx] for idx in predicted_classes]
    
    # 4. Save prediction results
    print(f"\n[Step 4/4] Save prediction results...")
    
    # Create DataFrame
    submission = pd.DataFrame({
        'name': file_names,
        'label': predicted_labels
    })
    
    # Save to CSV
    output_path = os.path.join(SUBMISSION_DIR, output_filename)
    submission.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n{'='*60}")
    print(f"Prediction completed!")
    print(f"Prediction file saved to: {output_path}")
    print(f"Number of predictions: {len(submission)}")
    print(f"\nPrediction preview:")
    print(submission.head(10))
    print(f"\nClass distribution:")
    print(submission['label'].value_counts().sort_index())
    print(f"{'='*60}")
    
    return submission


def batch_predict():
    """
    Batch predict test_a and test_b
    """
    print("\n" + "="*80)
    print("Batch prediction mode")
    print("="*80)
    
    # Select model
    model_name = f"cnn_{'sample' if USE_SAMPLE_DATA else 'full'}_best.h5"
    model_path = os.path.join(MODEL_SAVE_DIR, model_name)
    
    if not os.path.exists(model_path):
        print(f"\nWarning: Model not found: {model_path}")
        print("Please run train.py to train the model first!")
        
        # Try to find other available models
        available_models = [f for f in os.listdir(MODEL_SAVE_DIR) if f.endswith('.h5')]
        if available_models:
            print(f"\nAvailable models:")
            for i, m in enumerate(available_models, 1):
                print(f"{i}. {m}")
            print("\nPlease specify the correct model path in the code")
        return
    
    # Get timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Predict test_a
    print("\n" + "-"*80)
    print("Predicting test_a")
    print("-"*80)
    submission_a = predict_and_save(
        model_path=model_path,
        test_dir=TEST_A_DIR,
        output_filename=f'submission_test_a_{timestamp}.csv'
    )
    
    # Predict test_b
    print("\n" + "-"*80)
    print("Predicting test_b")
    print("-"*80)
    submission_b = predict_and_save(
        model_path=model_path,
        test_dir=TEST_B_DIR,
        output_filename=f'submission_test_b_{timestamp}.csv'
    )
    
    print("\n" + "="*80)
    print("Batch prediction completed!")
    print("="*80)
    print(f"\nSubmission files are saved in directory: {SUBMISSION_DIR}")


if __name__ == "__main__":
    # Set GPU memory growth (if using GPU)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # Execute batch prediction
    batch_predict()
