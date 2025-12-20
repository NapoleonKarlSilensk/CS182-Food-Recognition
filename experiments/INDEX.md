# 实验索引

本文件记录所有实验的概要信息。

| 实验ID | 实验名称 | 日期 | 验证准确率 | 训练时长 | 说明 |
|--------|----------|------|-----------|---------|------|
| [EXP002](EXP002_baseline_cnn_sample_20251220_032418/README.md) | baseline_cnn_sample | 2025-12-20 03:24:18 | 36.00% | 约20分钟 | Simplified 3-layer CNN, train_sample dataset |
| [EXP003](EXP003_baseline_cnn_full_20251220_032418/README.md) | baseline_cnn_full | 2025-12-20 03:24:18 | **57.86%** | 85.44 minutes | Simplified 3-layer CNN, full train dataset - **Current baseline** |
| EXP007 | yamnet_feature_extractor_sample | 2025-12-20 17:58:40 | 40.50% | 6.43 minutes | YAMNet transfer learning, sample data - **Failed: Lower than baseline** |
| EXP008 | yamnet_feature_extractor_full | 2025-12-20 19:09:18 | 54.93% | ~30 minutes | YAMNet transfer learning, full data - **Failed: 2.93% worse than baseline** |

## 实验结论

### YAMNet迁移学习尝试（EXP007-008）- 失败
- **结论**：YAMNet预训练模型在此任务上表现不佳，未能超越简单的CNN baseline
- **可能原因**：
  - YAMNet在AudioSet上预训练，但食物咀嚼声与AudioSet的音频类型差异较大
  - 使用tf.map_fn逐样本处理导致特征提取效率低下
  - 冻结YAMNet参数可能限制了模型适应新任务的能力
- **教训**：预训练模型并非万能，领域差异过大时可能不如针对性设计的简单模型
- **后续方向**：在baseline基础上探索数据增强、架构优化等方法
