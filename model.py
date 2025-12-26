# -*- coding: utf-8 -*-
"""
Model definitions: Define CNN and ResNet-like models for audio classification
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from config import *


def build_cnn(input_shape, num_classes=NUM_CLASSES):
    """
    Build a CNN model
    
    Args:
        input_shape: Shape of the input features (height, width, channels)
        num_classes: Number of classes for classification
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First convolutional block - reduce overfitting
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # global average pooling
        layers.GlobalAveragePooling2D(),
        
        # Simplified fully connected layers
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def residual_block(x, filters, strides=1, use_projection=False):
    """
    Build a residual block
    
    Args:
        x: Input tensor
        filters: Number of filters
        strides: Stride size
        use_projection: Whether to use projection shortcut
    
    Returns:
        Output tensor
    """
    shortcut = x
    
    # Main path
    x = layers.Conv2D(filters, (3, 3), strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Shortcut path with projection if needed
    if use_projection or strides != 1:
        shortcut = layers.Conv2D(filters, (1, 1), strides=strides, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Add shortcut
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x


def build_resnet(input_shape, num_classes=NUM_CLASSES):
    """
    Build an improved ResNet-like model for audio classification
    3 stages, stride convolution replaces some pooling
    
    Args:
        input_shape: Shape of the input features (height, width, channels)
        num_classes: Number of classes for classification
    
    Returns:
        Keras model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
    
    # Stage 1: 64 filters, 2 blocks
    x = residual_block(x, 64, strides=1, use_projection=True)
    x = residual_block(x, 64, strides=1)
    
    # Stage 2: 128 filters, 2 blocks (stride=2 downsampling)
    x = residual_block(x, 128, strides=2, use_projection=True)
    x = residual_block(x, 128, strides=1)
    
    # Stage 3: 256 filters, 2 blocks (stride=2 downsampling)
    x = residual_block(x, 256, strides=2, use_projection=True)
    x = residual_block(x, 256, strides=1)
    
    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Classifier - increase Dense layer capacity
    x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='ResNet_Audio')
    return model


def compile(model, learning_rate=LEARNING_RATE):
    """
    Compile the model
    
    Args:
        model: Keras model
        learning_rate: Learning rate
    
    Returns:
        Compiled Keras model
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def callbacks(model_name='audio_classifier'):
    """
    Get training callbacks
    
    Args:
        model_name: Name of the model
    
    Returns:
        List of callbacks
    """
    callbacks = [
        # Model checkpoint - save the best model
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_SAVE_DIR, f'{model_name}_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Early stopping - prevent overfitting
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Learning rate reduction
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        ),
        
        # TensorBoard
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(BASE_DIR, 'logs', model_name),
            histogram_freq=1
        )
    ]
    
    return callbacks


if __name__ == "__main__":
    # Test model building
    print("Testing CNN model...")
    input_shape = (N_MELS, 216, 1)  # Example input shape
    
    model_cnn = build_cnn_model(input_shape)
    model_cnn = compile_model(model_cnn)
    model_cnn.summary()
    
    print("\n" + "="*60)
    print("Testing ResNet-like model...")
    model_resnet = build_resnet_like_model(input_shape)
    model_resnet = compile_model(model_resnet)
    model_resnet.summary()
