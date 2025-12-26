# -*- coding: utf-8 -*-
"""
Data preprocessing module: Load and preprocess audio data
"""
import os
import numpy as np
import librosa
import librosa.display
from tqdm import tqdm
from config import *

class ADL:
    """Audio data loader and preprocessor"""
    
    def __init__(self, data_dir, sample_rate=SAMPLE_RATE, duration=DURATION):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.max_length = sample_rate * duration
        
    def load_audio(self, file_path):
        """load a single audio file"""
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            
            # Pad or truncate to fixed length
            if len(audio) < self.max_length:
                audio = np.pad(audio, (0, self.max_length - len(audio)), mode='constant')
            else:
                audio = audio[:self.max_length]
                
            return audio
        except Exception as e:
            print(f"Audio loading failed {file_path}: {str(e)}")
            return None
    
    def Mel(self, audio):
        """Extract Mel spectrogram features"""
        try:
            # Calculate Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=N_MELS,
                hop_length=HOP_LENGTH,
                n_fft=N_FFT
            )
            
            # transform to dB scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize to [0, 1]
            mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)
            
            return mel_spec_db
        except Exception as e:
            print(f"Feature extraction failed: {str(e)}")
            return None
    
    def load_data(self):
        """Load training data"""
        X = []
        y = []
        
        print(f"\nStart loading training data...")
        print(f"Data directory: {self.data_dir}")
        
        # Iterate through each class folder
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(self.data_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: Class folder does not exist - {class_dir}")
                continue
            
            # Get all audio files for this class
            audio_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
            
            print(f"Processing class [{class_idx+1}/{NUM_CLASSES}] {class_name}: {len(audio_files)} files")
            
            # Use progress bar to process each file
            for audio_file in tqdm(audio_files, desc=f"  {class_name}", leave=False):
                file_path = os.path.join(class_dir, audio_file)
                
                # Load audio
                audio = self.load_audio(file_path)
                if audio is None:
                    continue
                
                # Extract features
                mel_spec = self.Mel(audio)
                if mel_spec is None:
                    continue
                
                X.append(mel_spec)
                y.append(class_idx)
        
        X = np.array(X)
        y = np.array(y)
        
        # Add channel dimension (samples, height, width, channels)
        X = X[..., np.newaxis]
        
        print(f"\nData loading completed")
        print(f"Feature shape: {X.shape}")
        print(f"Label shape: {y.shape}")
        print(f"Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def load_test(self, test_dir):
        """Load test data"""
        X = []
        file_names = []
        
        print(f"\nStart loading test data...")
        print(f"Test directory: {test_dir}")
        
        # Get all test files
        audio_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.wav')])
        
        print(f"Number of test files: {len(audio_files)}")
        
        # Use progress bar to process each file
        for audio_file in tqdm(audio_files, desc="Loading test data"):
            file_path = os.path.join(test_dir, audio_file)
            
            # Load audio
            audio = self.load_audio(file_path)
            if audio is None:
                continue
            
            # Extract features
            mel_spec = self.Mel(audio)
            if mel_spec is None:
                continue
            
            X.append(mel_spec)
            file_names.append(audio_file)
        
        X = np.array(X)
        # Add channel dimension
        X = X[..., np.newaxis]
        
        print(f"\nTest data loading completed")
        print(f"Feature shape: {X.shape}")
        print(f"Number of files: {len(file_names)}")
        
        return X, file_names


def augment(audio):
    """
    Data augmentation: Apply random transformations to audio
    """
    augmented = []
    
    # Original audio
    augmented.append(audio)
    
    # 1. Add noise
    noise = np.random.randn(len(audio)) * 0.005
    augmented.append(audio + noise)
    
    # 2. Time stretching
    stretched = librosa.effects.time_stretch(audio, rate=0.9)
    if len(stretched) < len(audio):
        stretched = np.pad(stretched, (0, len(audio) - len(stretched)), mode='constant')
    else:
        stretched = stretched[:len(audio)]
    augmented.append(stretched)
    
    # 3. Pitch shifting
    pitched = librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=2)
    augmented.append(pitched)
    
    return augmented[:AUGMENTATION_FACTOR]


if __name__ == "__main__":
    # Test data loading
    loader = ADL(TRAIN_DATA_DIR)
    X, y = loader.load_training_data()
    print(f"\nData loading test successful")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
