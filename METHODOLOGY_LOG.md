# 方法论记录

## 当前方法 (Baseline) - 2025年12月20日

### 性能表现
- **验证准确率**: 57.86%
- **训练时长**: 85.44分钟
- **训练样本**: 5,600
- **验证样本**: 1,400

### 技术细节

#### 1. 特征提取
```python
特征类型: Mel频谱图
- 采样率: 22,050 Hz
- 音频长度: 5秒（固定）
- n_mels: 128
- n_fft: 2048 (默认)
- hop_length: 512 (默认)
- 输出形状: (128, 216, 1)
- 归一化: Min-Max归一化到 [0, 1]
```

**优点**:
- 实现简单
- 计算效率高
- 适合CNN处理

**缺点**:
- 单一特征，信息有限
- 未考虑时频域的局部不变性
- 缺少音色、节奏等信息

#### 2. 模型架构
```python
简化CNN (3层卷积 + 全连接)
├── Conv2D(32, 3x3) + ReLU + MaxPool(2x2) + Dropout(0.3)
├── Conv2D(64, 3x3) + ReLU + MaxPool(2x2) + Dropout(0.4)
├── Conv2D(128, 3x3) + ReLU + MaxPool(2x2) + Dropout(0.5)
├── GlobalAveragePooling2D
├── Dense(128) + L2(0.01) + ReLU + Dropout(0.5)
└── Dense(20) + Softmax

总参数: 111,764
```

**优点**:
- 参数少，训练快
- 避免了过拟合（相比之前的4层版本）
- GlobalAveragePooling减少参数

**缺点**:
- 感受野有限
- 缺乏skip connection，深层信息传递困难
- 无attention机制，无法聚焦关键特征
- 卷积核大小固定，无法捕获多尺度特征

#### 3. 训练策略
```python
优化器: Adam
- learning_rate: 0.001
- beta_1: 0.9 (默认)
- beta_2: 0.999 (默认)

正则化:
- L2正则化: 0.01 (仅Dense层)
- Dropout: 0.3 → 0.4 → 0.5 → 0.5
- 无BatchNormalization（移除后性能提升）

回调函数:
- EarlyStopping: patience=20, monitor='val_accuracy'
- ReduceLROnPlateau: patience=8, factor=0.5, min_lr=1e-6
- ModelCheckpoint: 保存最佳模型

数据分割: 80% train / 20% validation
批次大小: 32
```

**优点**:
- 学习率自适应调整
- 早停避免过拟合

**缺点**:
- **无数据增强**（关键缺陷）
- 单一学习率策略
- 无warmup
- 无label smoothing
- 无mixup/cutmix

#### 4. 数据处理
```python
音频加载:
- librosa.load(sr=22050)
- 填充/截断到固定长度

预处理:
- 无噪声注入
- 无时间拉伸
- 无音高变换
- 无响度归一化
```

**缺点**:
- **完全没有数据增强**（最大问题）
- 模型泛化能力受限
- 训练数据多样性不足

---

## 问题分析

### 主要瓶颈

1. **数据增强缺失** ⚠️ 
   - 当前7000样本对20个类别来说仍然偏少
   - 无augmentation导致模型泛化能力差
   - 某些类别样本极少（如chocolate: 178）

2. **特征单一** ⚠️
   - 仅使用Mel频谱，信息量有限
   - 未利用MFCC、Chroma、ZCR等互补特征

3. **模型容量不足** ⚠️
   - 3层CNN可能过于简单
   - 无残差连接，深度受限
   - 无attention机制，无法聚焦关键时频区域

4. **训练策略保守** ⚠️
   - 无cosine annealing等高级学习率调度
   - 无mixup等高级正则化

### 性能瓶颈证据

从验证结果看：
- burger: 79.69% ✓（样本多：372）
- ice-cream: 69.57% ✓（样本多：458）
- chocolate: 0% ✗（样本少：178）
- noodles: 0% ✗（样本少：251）

**结论**: 样本少的类别性能极差，急需数据增强

---

## 改进方向优先级

### 🔥 高优先级（预期提升15-25%）

1. **数据增强** (最重要)
   - SpecAugment (时频掩码)
   - 时间拉伸/压缩
   - 音高变换
   - 添加背景噪声
   - Mixup

2. **多特征融合**
   - Mel + MFCC
   - Mel + Chroma
   - 或concatenate多种特征

3. **改进模型架构**
   - ResNet-style残差连接
   - 注意力机制（CBAM/SE）
   - 多尺度卷积

### 🟡 中优先级（预期提升5-10%）

4. **高级训练策略**
   - Cosine annealing
   - Warmup
   - Label smoothing
   - Gradient accumulation

5. **集成学习**
   - 训练多个模型投票
   - 不同seed/架构的ensemble

### 🟢 低优先级（预期提升0-5%）

6. **超参数调优**
   - Grid search
   - 调整卷积核大小
   - 调整Dropout比例

---

## 下一步计划

根据优先级，建议首先实现：

### 方案A: 数据增强 + 改进CNN（推荐）
- SpecAugment
- 时间/音高变换
- ResNet-style架构
- **预期准确率**: 70-75%

### 方案B: 多特征融合
- Mel + MFCC双流网络
- Late fusion
- **预期准确率**: 65-70%

### 方案C: 注意力机制
- 基于当前CNN添加CBAM
- 关注关键时频区域
- **预期准确率**: 62-68%

---

## 实验记录

| 实验ID | 日期 | 方法 | Val Acc | 说明 |
|--------|------|------|---------|------|
| EXP-001 | 2025-12-20 | Baseline CNN (4层) | 7% | 严重过拟合 |
| EXP-002 | 2025-12-20 | Simplified CNN (3层) | 36% | train_sample |
| EXP-003 | 2025-12-20 | Simplified CNN (3层) | **57.86%** | 完整数据集 |
| EXP-004 | TBD | 待定 | TBD | 下一个实验 |

