# -*- coding: utf-8 -*-
"""
Configuration file: Set all parameters for the project
"""
import os

# ===== Basic path configuration =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TRAIN_SAMPLE_DIR = os.path.join(BASE_DIR, 'train_sample')
TEST_A_DIR = os.path.join(BASE_DIR, 'test_a')
TEST_B_DIR = os.path.join(BASE_DIR, 'test_b')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models')
SUBMISSION_DIR = os.path.join(BASE_DIR, 'submissions')

# ===== Training mode configuration =====
# Set to True to use train_sample for quick testing, set to False to use the full train dataset
USE_SAMPLE_DATA = False
#USE_SAMPLE_DATA = True

# Select training directory based on mode
TRAIN_DATA_DIR = TRAIN_SAMPLE_DIR if USE_SAMPLE_DATA else TRAIN_DIR

# ===== 音频处理参数 =====
SAMPLE_RATE = 16000  # 采样率 - 改为16kHz以减少计算量
DURATION = 5  # 统一音频长度（秒）
N_MELS = 128  # Mel频谱图的频带数量
HOP_LENGTH = 512  # STFT的跳跃长度
N_FFT = 2048  # FFT窗口大小
# 输出形状: (128, 157) for 16kHz 5秒音频

# ===== 数据增强参数 =====
USE_DATA_AUGMENTATION = True  # 是否使用数据增强
AUGMENTATION_FACTOR = 2  # 数据增强倍数

# ===== 模型参数 =====
NUM_CLASSES = 20  # 食物类别数量
BATCH_SIZE = 32  # 批次大小
EPOCHS = 100  # 训练轮次
LEARNING_RATE = 0.001  # 学习率
VALIDATION_SPLIT = 0.15  # 验证集比例

# ===== Food classification =====
CLASS_NAMES = [
    'aloe', 'burger', 'cabbage', 'candied_fruits', 'carrots',
    'chips', 'chocolate', 'drinks', 'fries', 'grapes',
    'gummies', 'ice-cream', 'jelly', 'noodles', 'pickles',
    'pizza', 'ribs', 'salmon', 'soup', 'wings'
]

# ===== Ensure directories exist =====
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"Configuration loaded successfully")
print(f"Training mode: {'Sample mode (train_sample)' if USE_SAMPLE_DATA else 'Full mode (train)'}")
print(f"Training data directory: {TRAIN_DATA_DIR}")
print(f"Number of classes: {NUM_CLASSES}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Number of epochs: {EPOCHS}")
print(f"{'='*60}\n")
