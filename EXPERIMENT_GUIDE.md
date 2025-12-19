# 实验管理系统使用指南

## 📚 系统简介

实验管理系统会自动为每次实验创建独立文件夹，保存所有训练数据、模型、可视化图表等，方便后续写论文和复现结果。

---

## 📁 实验文件夹结构

每次实验会创建如下结构：

```
experiments/
├── INDEX.md                              # 所有实验的索引列表
│
├── EXP002_baseline_cnn_sample_20251220_032418/
│   ├── README.md                         # 实验说明
│   ├── experiment_metadata.json          # 完整元数据
│   ├── plots/                            # 可视化图表
│   │   ├── training_curves.png           # 训练/验证曲线
│   │   └── confusion_matrix.png          # 混淆矩阵
│   ├── models/
│   │   └── best_model.h5                 # 最佳模型
│   ├── logs/
│   │   ├── config.json                   # 超参数配置
│   │   ├── model_summary.txt             # 模型架构
│   │   └── training_history.json         # 训练历史数据
│   ├── metrics/
│   │   ├── metrics.json                  # 性能指标
│   │   ├── confusion_matrix.npy          # 混淆矩阵原始数据
│   │   ├── classification_report.txt     # 分类报告
│   │   └── classification_report.json    # 分类报告(JSON)
│   └── predictions/
│       ├── test_a_predictions.csv        # test_a预测结果
│       └── test_b_predictions.csv        # test_b预测结果
│
└── EXP003_baseline_cnn_full_20251220_032418/
    └── ... (同上)
```

---

## 🚀 使用方法

### 方式1: 在训练脚本中使用（推荐）

```python
from experiment_logger import ExperimentLogger, create_experiments_index

# 1. 创建实验记录器
logger = ExperimentLogger('yamnet_transfer_learning')

# 2. 记录配置
config = {
    'model': 'YAMNet',
    'pretrained': True,
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.0001,
    'dataset': 'train_full'
}
logger.log_config(config)

# 3. 记录模型架构
logger.log_model_summary(model)

# 4. 训练模型
history = model.fit(...)

# 5. 记录训练历史（自动绘制曲线）
logger.log_training_history(history)

# 6. 评估并记录指标
test_loss, test_acc = model.evaluate(X_test, y_test)
metrics = {
    'validation_accuracy': test_acc,
    'validation_loss': test_loss,
    'training_time': '45 minutes'
}
logger.log_metrics(metrics)

# 7. 记录混淆矩阵
from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_true, y_pred)
logger.log_confusion_matrix(cm, class_names)

# 8. 保存模型
logger.save_model(model, 'best_model.h5')

# 9. 保存预测结果
predictions_df = pd.DataFrame({'name': files, 'label': pred_labels})
logger.save_predictions(predictions_df, 'test_a_predictions.csv')

# 10. 完成实验
logger.finalize(final_metrics=metrics)

# 11. 更新实验索引
create_experiments_index()
```

### 方式2: 归档已有实验

```python
from experiment_logger import ExperimentLogger
import shutil

logger = ExperimentLogger('my_past_experiment')

# 记录配置和指标
logger.log_config(my_config_dict)
logger.log_metrics(my_metrics_dict)

# 复制已有文件
shutil.copy('old_model.h5', 
            os.path.join(logger.folders['models'], 'best_model.h5'))
shutil.copy('training_plot.png',
            os.path.join(logger.folders['plots'], 'training_curves.png'))

logger.finalize()
```

---

## 📊 查看实验结果

### 方法1: 浏览器查看
打开 `experiments/INDEX.md` 查看所有实验概览

### 方法2: 查看单个实验
进入实验文件夹，打开 `README.md`

### 方法3: 编程读取
```python
import json

# 读取实验元数据
with open('experiments/EXP003_baseline_cnn_full_xxx/experiment_metadata.json') as f:
    metadata = json.load(f)
    print(metadata['metrics'])

# 读取训练历史
with open('experiments/EXP003_baseline_cnn_full_xxx/logs/training_history.json') as f:
    history = json.load(f)
    print(history['val_accuracy'])
```

---

## 🎯 已归档的实验

### EXP002: Baseline CNN (train_sample)
- **准确率**: 36%
- **数据集**: train_sample (1000样本)
- **说明**: 简化3层CNN，用于快速验证

### EXP003: Baseline CNN (full train) ⭐ 当前基线
- **准确率**: 57.86%
- **数据集**: train (7000样本)
- **说明**: 从头训练的3层CNN，当前baseline

---

## 📝 写论文时的使用建议

### 1. 实验对比表格
从 `INDEX.md` 直接复制表格到论文

### 2. 训练曲线图
使用 `plots/training_curves.png`（300 DPI，适合论文）

### 3. 混淆矩阵
使用 `plots/confusion_matrix.png`

### 4. 性能指标
从 `metrics/metrics.json` 读取精确数值

### 5. 超参数表格
从 `logs/config.json` 获取所有配置

### 6. 实验复现
所有配置和代码都保存在实验文件夹，可完全复现

---

## 🔧 自定义和扩展

### 添加自定义可视化
```python
import matplotlib.pyplot as plt

# 绘制你的图表
plt.figure()
# ... 绘图代码 ...

# 保存到实验文件夹
plot_path = os.path.join(logger.folders['plots'], 'my_custom_plot.png')
plt.savefig(plot_path, dpi=300)
plt.close()
```

### 记录额外信息
```python
# 可以在metadata中添加任何信息
logger.metadata['my_custom_field'] = 'custom value'

# 保存自定义文件
import pickle
with open(os.path.join(logger.folders['metrics'], 'custom_data.pkl'), 'wb') as f:
    pickle.dump(my_data, f)
```

---

## 💡 最佳实践

1. **每次重要实验都使用记录器** - 养成习惯
2. **实验名称要描述性** - 如 `yamnet_lr0001_augmented`
3. **及时完成实验** - 调用 `finalize()` 生成完整报告
4. **定期更新索引** - 运行 `create_experiments_index()`
5. **添加实验备注** - 在metadata中记录实验想法和观察

---

## 🎓 写论文checklist

从实验文件夹获取：
- [ ] 训练/验证曲线图
- [ ] 混淆矩阵
- [ ] 性能指标表格
- [ ] 超参数配置表格
- [ ] 模型架构说明
- [ ] 实验对比表
- [ ] 时间成本分析

所有这些都已自动保存在实验文件夹中！
