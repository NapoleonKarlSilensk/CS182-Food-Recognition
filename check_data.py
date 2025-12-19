# -*- coding: utf-8 -*-
"""
数据检查脚本：验证数据加载是否正确
"""
import numpy as np
import matplotlib.pyplot as plt
from config import *
from data_preprocessing import ADL

# 加载少量数据检查
loader = ADL(TRAIN_SAMPLE_DIR)

# 测试加载一个文件
test_file = "d:\\shanghaitech\\CS182proj-me\\train_sample\\aloe\\24EJ22XBZ5.wav"
audio = loader.load_audio(test_file)
mel_spec = loader.Mel(audio)

print("音频形状:", audio.shape)
print("音频范围:", audio.min(), "到", audio.max())
print("Mel频谱图形状:", mel_spec.shape)
print("Mel频谱图范围:", mel_spec.min(), "到", mel_spec.max())
print("Mel频谱图均值:", mel_spec.mean())
print("Mel频谱图标准差:", mel_spec.std())

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(audio[:1000])
axes[0].set_title('Audio waveform (first 1000 samples)')
axes[1].imshow(mel_spec, aspect='auto', origin='lower', cmap='viridis')
axes[1].set_title('Mel Spectrogram')
plt.tight_layout()
plt.savefig('data_check.png')
print("\n可视化已保存到 data_check.png")
