# EXP002: baseline_cnn_sample

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
  "dataset": "train_sample",
  "train_samples": 800,
  "val_samples": 200,
  "epochs": 200,
  "batch_size": 32,
  "learning_rate": 0.001,
  "optimizer": "Adam",
  "callbacks": [
    "EarlyStopping(patience=20)",
    "ReduceLROnPlateau(patience=8)"
  ]
}
```

## 性能指标

- **validation_accuracy**: 0.36
- **validation_loss**: 2.1951
- **training_time**: 约20分钟
- **best_epoch**: 176
- **total_params**: 111764

## 文件结构

```
EXP002_baseline_cnn_sample_20251220_032418/
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
