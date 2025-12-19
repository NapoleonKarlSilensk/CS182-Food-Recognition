# EXP003: baseline_cnn_full

**实验日期**: 2025-12-20 03:24:18

## 配置参数

```json
{
  "model_type": "CNN",
  "architecture": "3-layer Simplified CNN",
  "conv_layers": 3,
  "filters": [
    32,
    64,
    128
  ],
  "dense_units": 128,
  "dropout_rates": [
    0.3,
    0.4,
    0.5,
    0.5
  ],
  "l2_regularization": 0.01,
  "dataset": "train (full)",
  "train_samples": 5600,
  "val_samples": 1400,
  "total_samples": 7000,
  "num_classes": 20,
  "epochs": 200,
  "batch_size": 32,
  "learning_rate": 0.001,
  "optimizer": "Adam",
  "callbacks": [
    "EarlyStopping(patience=20)",
    "ReduceLROnPlateau(patience=8)",
    "ModelCheckpoint"
  ],
  "early_stopping_epoch": 199
}
```

## 性能指标

- **validation_accuracy**: 0.5786
- **validation_loss**: 1.4183
- **training_accuracy**: N/A
- **training_time**: 85.44 minutes
- **best_epoch**: 199
- **total_params**: 111764
- **trainable_params**: 111764
- **note**: Current baseline - 从头训练的3层CNN

## 文件结构

```
EXP003_baseline_cnn_full_20251220_032418/
├── README.md                    # 本文件
├── experiment_metadata.json     # 完整元数据
├── plots/
│   ├── training_curves.png      # 训练曲线
│   └── confusion_matrix.png     # 混淆矩阵
├── models/
│   └── best_model.h5            # 最佳模型
├── logs/
│   ├── config.json              # 配置参数
│   ├── model_summary.txt        # 模型架构
│   └── training_history.json    # 训练历史
├── metrics/
│   ├── metrics.json             # 性能指标
│   ├── confusion_matrix.npy     # 混淆矩阵数据
│   ├── classification_report.txt
│   └── classification_report.json
└── predictions/
    └── predictions.csv          # 预测结果
```
