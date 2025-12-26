# -*- coding: utf-8 -*-
"""
Optimized CNN training script
"""
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split

from config import *
from data_preprocessing import ADL
from experiment_logger import ExperimentLogger


def build_optimized_cnn(input_shape, num_classes=NUM_CLASSES):
    """
    Build optimized CNN with improvements:
    1. Batch Normalization for training stability
    2. More conv layers (4 layers vs 3)
    3. Larger filters (64, 128, 256, 512)
    4. Better regularization
    """
    model = tf.keras.Sequential([
        # Input
        tf.keras.layers.Input(shape=input_shape),
        
        # Block 1: 64 filters
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        
        # Block 2: 128 filters
        tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        
        # Block 3: 256 filters
        tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),
        
        # Block 4: 512 filters
        tf.keras.layers.Conv2D(512, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        
        # Dense layers
        tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.5),
        
        # Output
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def train_optimized_cnn(use_sample_data=False):
    """
    Train optimized CNN with same config as ResNet for fair comparison
    """
    print("\n" + "="*80)
    print("Optimized CNN Training - Fair Comparison")
    print("="*80)
    
    # Determine data directory
    if use_sample_data:
        data_dir = TRAIN_SAMPLE_DIR
        exp_prefix = "cnn_optimized_sample"
    else:
        data_dir = TRAIN_DIR
        exp_prefix = "cnn_optimized_full"
    
    print(f"Data directory: {data_dir}")
    print(f"Sampling rate: {SAMPLE_RATE}Hz (same as ResNet)")
    print(f"Validation split: {VALIDATION_SPLIT*100}% (same as ResNet)")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max epochs: {EPOCHS}")
    
    # Create experiment logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{exp_prefix}_{timestamp}"
    logger = ExperimentLogger(exp_name)
    
    # [Step 1/7] Load data
    print(f"\n[Step 1/7] Loading training data...")
    loader = ADL(data_dir, sample_rate=SAMPLE_RATE, duration=DURATION)
    X, y = loader.load_data()
    class_names = CLASS_NAMES
    
    print(f"[OK] Data loaded successfully")
    print(f"  Features shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")
    print(f"  Number of classes: {len(class_names)}")
    
    # [Step 2/7] Split training and validation sets (15% stratified)
    print(f"\n[Step 2/7] Splitting training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=VALIDATION_SPLIT,
        stratify=y,
        random_state=42
    )
    
    print(f"[OK] Data split completed")
    print(f"  Training set: {X_train.shape[0]} samples ({(1-VALIDATION_SPLIT)*100:.0f}%)")
    print(f"  Validation set: {X_val.shape[0]} samples ({VALIDATION_SPLIT*100:.0f}%)")
    
    # [Step 3/7] Build model
    print(f"\n[Step 3/7] Building optimized CNN model...")
    input_shape = X_train.shape[1:]
    model = build_optimized_cnn(input_shape=input_shape, num_classes=NUM_CLASSES)
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"[OK] Model built successfully")
    print(f"  Total parameters: {model.count_params():,}")
    
    model.summary()
    
    # [Step 4/7] Log experiment config
    print(f"\n[Step 4/7] Logging experiment configuration...")
    
    config_dict = {
        'model_type': 'Optimized CNN',
        'architecture': '4-block CNN with BatchNorm',
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
        'optimizations': [
            '16kHz sampling (vs 22kHz in old CNN)',
            '15% validation split (vs 20% in old CNN)',
            'Early Stopping patience=15 (vs 20 in old CNN)',
            'ReduceLR patience=5, factor=0.5 (vs patience=8 in old CNN)',
            'Batch Normalization added',
            '4 blocks (vs 3 in old CNN)',
            'Larger filters: 64→128→256→512 (vs 32→64→128)',
        ]
    }
    
    logger.log_config(config_dict)
    logger.log_model(model)
    print(f"[OK] Experiment config and model logged")
    
    # [Step 5/7] Setup callbacks (same as ResNet)
    print(f"\n[Step 5/7] Setup callbacks...")
    callbacks = [
        # Early stopping - patience=15 (same as ResNet)
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            verbose=1,
            restore_best_weights=True
        ),
        # Learning rate reduction - patience=5, factor=0.5 (same as ResNet)
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
    
    print(f"[OK] Callbacks configured:")
    print(f"  - EarlyStopping: patience=15 (same as ResNet)")
    print(f"  - ReduceLROnPlateau: patience=5, factor=0.5 (same as ResNet)")
    
    # [Step 6/7] Start training
    print(f"\n[Step 6/7] Starting training...")
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
    
    print(f"\n[OK] Training completed! Time taken: {training_time:.2f} minutes")
    
    # Log training history
    logger.log_history(history)
    
    # [Step 7/7] Evaluate model
    print(f"\n[Step 7/7] Evaluating model...")
    
    # Evaluate on validation set
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"[OK] Validation accuracy: {val_acc*100:.2f}%")
    print(f"     Validation loss: {val_loss:.4f}")
    
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
        'comparison_note': 'Optimized CNN with same config as ResNet for fair comparison'
    }
    logger.log_metrics(final_metrics)
    
    print(f"\n{'='*80}")
    print(f"Training completed! Experiment results saved to:")
    print(f"{logger.exp_folder}")
    print(f"{'='*80}")
    
    # Print comparison hint
    print(f"\nComparison Reference:")
    print(f"  Old CNN:                57.86% (22kHz, 20% val split)")
    print(f"  Optimized CNN (this):   {val_acc*100:.2f}% (16kHz, 15% val split)")
    print(f"  ResNet:                 96.10% (16kHz, 15% val split)")
    
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
    
    # Train optimized CNN
    model, history, logger = train_optimized_cnn(use_sample_data=False)
