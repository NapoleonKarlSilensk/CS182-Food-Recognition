# -*- coding: utf-8 -*-
"""
Training script: Train audio classification models
"""
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import time

from config import *
from data_preprocessing import ADL
from model import build_cnn_model, build_resnet_like_model, compile_model, get_callbacks


def plot_training_history(history, model_name):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(MODEL_SAVE_DIR, f'{model_name}_training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining history plot saved to: {save_path}")
    plt.close()


def train_model(model_type='cnn', epochs=EPOCHS):
    """
    Main function to train the model
    
    Args:
        model_type: Type of model ('cnn' or 'resnet')
        epochs: Number of training epochs
    """
    print("\n" + "="*80)
    print(f"Start training - Model type: {model_type.upper()}")
    print("="*80)
    
    # 1. Load data
    print("\n[Step 1/5] Loading training data...")
    loader = ADL(TRAIN_DATA_DIR)
    X, y = loader.load_training_data()
    
    # 2. Split training and validation sets
    print(f"\n[Step 2/5] Splitting dataset (Validation split: {VALIDATION_SPLIT})...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=VALIDATION_SPLIT, 
        random_state=42,
        stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    
    # 3. Build model
    print(f"\n[Step 3/5] Building {model_type.upper()} model...")
    input_shape = X_train.shape[1:]  # (height, width, channels)
    
    if model_type.lower() == 'cnn':
        model = build_cnn_model(input_shape)
    elif model_type.lower() == 'resnet':
        model = build_resnet_like_model(input_shape)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model = compile_model(model, LEARNING_RATE)
    
    print("\nModel architecture:")
    model.summary()
    
    # 4. Train model
    print(f"\n[Step 4/5] Starting training (Epochs: {epochs})...")
    model_name = f"{model_type}_{'sample' if USE_SAMPLE_DATA else 'full'}"
    
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(model_name),
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    print(f"\nTraining completed! Time taken: {training_time/60:.2f} minutes")
    
    # 5. Evaluate model
    print(f"\n[Step 5/5] Evaluating model...")
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"\n{'='*60}")
    print(f"Final validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Final validation loss: {val_loss:.4f}")
    print(f"{'='*60}")
    
    # Plot training history
    plot_training_history(history, model_name)
    
    # 保存最终模型
    final_model_path = os.path.join(MODEL_SAVE_DIR, f'{model_name}_final.h5')
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Save training information
    info_path = os.path.join(MODEL_SAVE_DIR, f'{model_name}_info.txt')
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(f"Model type: {model_type}\n")
        f.write(f"Training mode: {'Sample mode' if USE_SAMPLE_DATA else 'Full mode'}\n")
        f.write(f"Number of training samples: {X_train.shape[0]}\n")
        f.write(f"Number of validation samples: {X_val.shape[0]}\n")
        f.write(f"Number of epochs: {epochs}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Learning rate: {LEARNING_RATE}\n")
        f.write(f"Training time: {training_time/60:.2f} minutes\n")
        f.write(f"Final validation accuracy: {val_accuracy*100:.2f}%\n")
        f.write(f"Final validation loss: {val_loss:.4f}\n")
    
    print(f"Training information saved to: {info_path}")
    
    return model, history


if __name__ == "__main__":
    # Set GPU memory growth (if using GPU)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Detected {len(gpus)} GPUs, enabled memory growth mode")
        except RuntimeError as e:
            print(e)
    
    # Train CNN model
    print("\n" + "="*80)
    print("Training CNN model")
    print("="*80)
    model_cnn, history_cnn = train_model(model_type='cnn', epochs=EPOCHS)
    
    print("\n\n" + "="*80)
    print("All training tasks completed!")
    print("="*80)
    
    # Optional: Train ResNet model
    # print("\n\n" + "="*80)
    # print("Training ResNet model")
    # print("="*80)
    # model_resnet, history_resnet = train_model(model_type='resnet', epochs=EPOCHS)
