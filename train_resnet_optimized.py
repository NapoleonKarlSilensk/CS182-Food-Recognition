# -*- coding: utf-8 -*-
"""
Optimized ResNet training script
Based on new configuration: 16kHz sampling rate, 3-stage ResNet, 15% validation split
"""
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split

from config import *
from data_preprocessing import ADL
from model import build_resnet, compile
from experiment_logger import ExperimentLogger


def train_resnet_optimized(use_sample_data=False):
    """
    Train optimized ResNet model
    
    Configuration improvements:
    - 16kHz sampling rate (vs 22kHz)
    - 3-stage ResNet (vs 4-stage)
    - 15% validation split (vs 20%)
    - Early Stopping patience=15 (vs 20)
    - ReduceLROnPlateau patience=5 (vs 8)
    """
    print("\n" + "="*80)
    print("Optimized ResNet training")
    print("="*80)
    
    # Determine data directory
    if use_sample_data:
        data_dir = TRAIN_SAMPLE_DIR
        exp_prefix = "resnet_optimized_sample"
    else:
        data_dir = TRAIN_DIR
        exp_prefix = "resnet_optimized_full"
    
    print(f"data directory: {data_dir}")
    print(f"sampling rate: {SAMPLE_RATE}Hz")
    print(f"validation split: {VALIDATION_SPLIT*100}%")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max epochs: {EPOCHS}")
    
    # Create experiment logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{exp_prefix}_{timestamp}"
    logger = ExperimentLogger(exp_name)
    
    # [Step 1/6] Load data
    print(f"\n[Step 1/6] Loading training data...")
    loader = ADL(data_dir, sample_rate=SAMPLE_RATE, duration=DURATION)
    X, y = loader.load_data()
    class_names = CLASS_NAMES
    
    print(f"[OK] Data loaded successfully")
    print(f"  Features shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")
    print(f"  Number of classes: {len(class_names)}")
    
    # [Step 2/6] Split training and validation sets (15% stratified)
    print(f"\n[Step 2/6] Splitting training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=VALIDATION_SPLIT,
        stratify=y,
        random_state=42
    )
    
    print(f"[OK] Data split completed")
    print(f"  Training set: {X_train.shape[0]} samples ({(1-VALIDATION_SPLIT)*100:.0f}%)")
    print(f"  Validation set: {X_val.shape[0]} samples ({VALIDATION_SPLIT*100:.0f}%)")
    
    # [Step 3/6] Build model
    print(f"\n[Step 3/6] Building optimized ResNet model...")
    input_shape = X_train.shape[1:]
    model = build_resnet(input_shape=input_shape, num_classes=NUM_CLASSES)
    model = compile(model, learning_rate=LEARNING_RATE)
    
    print("[OK] Model built successfully")
    model.summary()
    
    # Save model architecture
    logger.log_model(model)
    
    # Save configuration
    config_dict = {
        'sample_rate': SAMPLE_RATE,
        'duration': DURATION,
        'n_mels': N_MELS,
        'n_fft': N_FFT,
        'hop_length': HOP_LENGTH,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'validation_split': VALIDATION_SPLIT,
        'use_sample_data': use_sample_data,
        'total_samples': len(X),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'num_classes': NUM_CLASSES,
        'input_shape': input_shape,
        'model_name': 'ResNet_Optimized_3Stage',
    }
    logger.log_config(config_dict)
    
    # [Step 4/6] Set training callbacks
    print(f"\n[Step 4/6] Setting training callbacks...")
    
    callbacks = [
        # Early stopping - patience=15 (more aggressive)
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            verbose=1,
            restore_best_weights=True
        ),
        # learning rate decline - patience=5, factor=0.5
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(logger.folders['models'], 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print("[OK] Callbacks set successfully")
    print(f"  - EarlyStopping: patience=15")
    print(f"  - ReduceLROnPlateau: patience=5, factor=0.5")
    
    # [Step 5/6] Start training
    print(f"\n[Step 5/6] Starting training...")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    
    start_time = datetime.now()
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds() / 60
    
    print(f"\nTraining completed! Time taken: {training_time:.2f} minutes")
    
    # Save training history
    logger.log_history(history)
    
    # [Step 6/6] Evaluate model
    print(f"\n[Step 6/6] Evaluating model...")
    
    # Evaluate on validation set
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation accuracy: {val_acc*100:.2f}%")
    print(f"Validation loss: {val_loss:.4f}")
    
    # Predict and generate confusion matrix
    y_pred = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Generate classification report and confusion matrix
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_classes, target_names=class_names))
    
    cm = confusion_matrix(y_val, y_pred_classes)
    logger.log_cm(cm, class_names)
    
    # Save classification report
    report_str = classification_report(y_val, y_pred_classes, target_names=class_names)
    report_path = os.path.join(logger.folders['metrics'], 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_str)
    print(f"[OK] Classification report saved: {report_path}")
    
    # Save final metrics
    final_metrics = {
        'val_accuracy': float(val_acc),
        'val_loss': float(val_loss),
        'training_time_minutes': float(training_time),
        'total_epochs': len(history.history['loss']),
        'best_epoch': int(np.argmax(history.history['val_accuracy'])) + 1,
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
    }
    logger.log_metrics(final_metrics)
    
    print(f"\n{'='*80}")
    print(f"Training completed! Experiment results saved to: {logger.exp_folder}")
    print(f"{'='*80}")
    
    return model, history, logger


if __name__ == "__main__":
    # Set GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # Use full data by default for ensemble training
    use_sample = False
    print("\n[INFO] Training with full dataset for ensemble model...")
    
    # Start training
    model, history, logger = train_resnet_optimized(use_sample_data=use_sample)
