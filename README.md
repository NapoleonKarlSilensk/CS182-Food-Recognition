# 食物音频分类项目的readme

基于深度学习的音频食物分类系统，通过识别食用声音判断食物类型。

## 项目结构

```
CS182proj-me/
├── config.py                    # 基础文件配置
├── data_preprocessing.py        # 数据预处理
├── model.py                     # 模型定义
├── train.py                     # 训练脚本
├── predict.py                   # 预测脚本
├── README.md                    # 没什么好说的
├── requirements.txt             # Python依赖
├── train/                       # 完整训练数据（20个类别文件夹）
├── train_sample/                # 样本训练数据（用于快速测试）
├── test_a/                      # 测试集A
├── test_b/                      # 测试集B
├── models/                      # 保存训练好的模型
└── submissions/                 # 预测结果提交文件
```

## 使用指导

### 1. 环境要求

- Python
- TensorFlow
- librosa
- numpy
- pandas
- scikit-learn
- matplotlib
- tqdm

### 2. 训练模型

在 `config.py` 文件中修改：

```python
USE_SAMPLE_DATA = True   # 使用 train_sample 快速测试（推荐首次运行）
# 或
USE_SAMPLE_DATA = False  # 使用完整 train 数据集（正式训练）
```

运行训练：

```bash
python train.py
```

### 3. 预测测试集

训练完成后，运行预测脚本：

```bash
python predict.py
```

预测结果将保存在 `submissions/` 目录下，格式符合天池要求：

```
name,label
DP2R8P7KJK.wav,cabbage
3ITH5DYEUI.wav,grapes
O91I9CHZBZ.wav,noodles
...
```

生成的文件名：

- `submission_test_a_<时间戳>.csv`
- `submission_test_b_<时间戳>.csv`

## 配置说明

### config.py 主要参数

```python
# 训练模式
USE_SAMPLE_DATA = True  # True=样本模式, False=完整模式

# 音频处理参数
SAMPLE_RATE = 22050     # 采样率
DURATION = 5            # 音频长度（秒）
N_MELS = 128            # Mel频谱图频带数

# 模型训练参数
BATCH_SIZE = 32         # 批次大小
EPOCHS = 50             # 训练轮数
LEARNING_RATE = 0.001   # 学习率
VALIDATION_SPLIT = 0.2  # 验证集比例
```

## 食物类别（20类）

1. aloe（芦荟）
2. burger（汉堡）
3. cabbage（卷心菜）
4. candied_fruits（蜜饯）
5. carrots（胡萝卜）
6. chips（薯片）
7. chocolate（巧克力）
8. drinks（饮料）
9. fries（薯条）
10. grapes（葡萄）
11. gummies（软糖）
12. ice-cream（冰淇淋）
13. jelly（果冻）
14. noodles（面条）
15. pickles（泡菜）
16. pizza（披萨）
17. ribs（排骨）
18. salmon（三文鱼）
19. soup（汤）
20. wings（鸡翅）

## 模型架构

### CNN模型

- 4个卷积块（Conv2D + BatchNorm + MaxPool + Dropout）
- 全局平均池化
- 2个全连接层
- Softmax输出层

### 训练策略

- 优化器：Adam
- 损失函数：Sparse Categorical Crossentropy
- 回调函数：
  - ModelCheckpoint（保存最佳模型）
  - EarlyStopping（早停）
  - ReduceLROnPlateau（学习率衰减）
  - TensorBoard（训练可视化）

## 使用流程

### 方案1：快速测试（推荐首次）

```bash
# 1. 设置为样本模式
# 在 config.py 中设置: USE_SAMPLE_DATA = True

# 2. 训练模型（几分钟完成）
python train.py

# 3. 预测测试集
python predict.py
```

### 方案2：正式训练

```bash
# 1. 设置为完整模式
# 在 config.py 中设置: USE_SAMPLE_DATA = False

# 2. 训练模型（需要较长时间）
python train.py

# 3. 预测测试集
python predict.py
```

## 输出文件说明

### 训练输出（models/目录）

- `cnn_sample_best.h5` - 最佳模型（样本模式）
- `cnn_full_best.h5` - 最佳模型（完整模式）
- `cnn_*_final.h5` - 最终模型
- `cnn_*_info.txt` - 训练信息
- `cnn_*_training_history.png` - 训练曲线图

### 预测输出（submissions/目录）

- `submission_test_a_*.csv` - test_a预测结果
- `submission_test_b_*.csv` - test_b预测结果

CSV格式示例：

```
name,label
DP2R8P7KJK.wav,cabbage
3ITH5DYEUI.wav,grapes
O91I9CHZBZ.wav,noodles
AH2AXWMRAP.wav,ice-cream
...
```

## 性能优化建议

1. **GPU加速**：如果有NVIDIA GPU，确保安装了CUDA和cuDNN
2. **调整参数**：
   - 增加 `EPOCHS` 提高训练充分性
   - 调整 `BATCH_SIZE`（根据内存）
   - 修改 `LEARNING_RATE` 优化收敛
3. **数据增强**：在 `data_preprocessing.py` 中启用数据增强
4. **模型选择**：尝试ResNet模型（在 `train.py` 末尾取消注释）

## 常见问题

### Q1: 训练太慢怎么办？

A: 先使用 `USE_SAMPLE_DATA = True` 快速测试，确认代码无误后再用完整数据集。

### Q2: 内存不足？

A: 减小 `BATCH_SIZE` 或使用样本模式训练。

### Q3: 如何提高准确率？

A:

- 增加训练轮数（EPOCHS）
- 使用完整训练集
- 启用数据增强
- 尝试不同的模型架构

### Q4: 找不到训练好的模型？

A: 检查 `models/` 目录，确保先运行了 `train.py`。
