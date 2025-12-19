# -*- coding: utf-8 -*-
"""
验证脚本：使用训练好的模型验证train_sample数据集的准确性
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from config import *
from data_preprocessing import ADL


def plot_confusion_matrix(cm, class_names, save_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': '样本数量'})
    plt.title('混淆矩阵 - Train Sample验证结果', fontsize=16, pad=20)
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n混淆矩阵已保存到: {save_path}")
    plt.close()


def validate_model(model_path, data_dir=TRAIN_SAMPLE_DIR):
    """
    验证模型在训练数据集上的表现
    
    Args:
        model_path: 模型文件路径
        data_dir: 验证数据目录（默认使用train_sample）
    """
    print("\n" + "="*80)
    print("模型验证 - 检测训练数据集准确性")
    print("="*80)
    print(f"模型路径: {model_path}")
    print(f"验证数据: {data_dir}")
    
    # 1. 加载模型
    print("\n[步骤 1/5] 加载模型...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model = keras.models.load_model(model_path)
    print("模型加载成功!")
    
    # 2. 加载验证数据
    print(f"\n[步骤 2/5] 加载验证数据...")
    loader = ADL(data_dir)
    X, y_true = loader.load_training_data()
    
    print(f"验证样本数: {len(y_true)}")
    print(f"类别分布: {dict(zip(CLASS_NAMES, np.bincount(y_true)))}")
    
    # 3. 进行预测
    print(f"\n[步骤 3/5] 进行预测...")
    predictions = model.predict(X, batch_size=BATCH_SIZE, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    
    # 4. 计算准确率
    print(f"\n[步骤 4/5] 计算准确率...")
    
    # 整体准确率
    accuracy = np.mean(y_pred == y_true)
    
    # 每个类别的准确率
    class_accuracies = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_mask = (y_true == class_idx)
        if np.sum(class_mask) > 0:
            class_acc = np.mean(y_pred[class_mask] == y_true[class_mask])
            class_accuracies[class_name] = class_acc
        else:
            class_accuracies[class_name] = 0.0
    
    # 5. 分析错误
    print(f"\n[步骤 5/5] 分析错误分类...")
    
    # 找出所有错误分类的样本
    errors = []
    for i in range(len(y_true)):
        if y_pred[i] != y_true[i]:
            errors.append({
                'index': i,
                'true_label': CLASS_NAMES[y_true[i]],
                'pred_label': CLASS_NAMES[y_pred[i]],
                'confidence': predictions[i][y_pred[i]] * 100
            })
    
    # 打印结果
    print("\n" + "="*80)
    print("验证结果汇总")
    print("="*80)
    
    print(f"\n整体准确率: {accuracy*100:.2f}% ({np.sum(y_pred == y_true)}/{len(y_true)})")
    print(f"错误分类数: {len(errors)}")
    print(f"正确分类数: {np.sum(y_pred == y_true)}")
    
    print("\n" + "-"*80)
    print("各类别准确率:")
    print("-"*80)
    
    # 按准确率排序
    sorted_classes = sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True)
    
    for class_name, acc in sorted_classes:
        class_idx = CLASS_NAMES.index(class_name)
        total = np.sum(y_true == class_idx)
        correct = np.sum((y_true == class_idx) & (y_pred == class_idx))
        
        status = "✓" if acc > 0.8 else ("⚠" if acc > 0.6 else "✗")
        print(f"{status} {class_name:20s}: {acc*100:6.2f}% ({correct:3d}/{total:3d})")
    
    # 显示错误分类案例
    if errors:
        print("\n" + "-"*80)
        print(f"错误分类案例 (显示前20个):")
        print("-"*80)
        
        for i, error in enumerate(errors[:20], 1):
            print(f"{i:2d}. 样本#{error['index']:4d}: "
                  f"真实={error['true_label']:15s} -> "
                  f"预测={error['pred_label']:15s} "
                  f"(置信度: {error['confidence']:.1f}%)")
        
        if len(errors) > 20:
            print(f"\n... 还有 {len(errors) - 20} 个错误分类")
    
    # 生成分类报告
    print("\n" + "-"*80)
    print("详细分类报告:")
    print("-"*80)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
    
    # 生成混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    cm_save_path = os.path.join(MODEL_SAVE_DIR, 'confusion_matrix_validation.png')
    plot_confusion_matrix(cm, CLASS_NAMES, cm_save_path)
    
    # 保存详细结果
    result_path = os.path.join(MODEL_SAVE_DIR, 'validation_results.txt')
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("模型验证结果\n")
        f.write("="*80 + "\n\n")
        f.write(f"模型路径: {model_path}\n")
        f.write(f"验证数据: {data_dir}\n")
        f.write(f"验证样本数: {len(y_true)}\n\n")
        
        f.write(f"整体准确率: {accuracy*100:.2f}%\n")
        f.write(f"正确分类: {np.sum(y_pred == y_true)}/{len(y_true)}\n")
        f.write(f"错误分类: {len(errors)}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("各类别准确率:\n")
        f.write("-"*80 + "\n")
        for class_name, acc in sorted_classes:
            class_idx = CLASS_NAMES.index(class_name)
            total = np.sum(y_true == class_idx)
            correct = np.sum((y_true == class_idx) & (y_pred == class_idx))
            f.write(f"{class_name:20s}: {acc*100:6.2f}% ({correct:3d}/{total:3d})\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("分类报告:\n")
        f.write("-"*80 + "\n")
        f.write(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
        
        if errors:
            f.write("\n" + "-"*80 + "\n")
            f.write("所有错误分类案例:\n")
            f.write("-"*80 + "\n")
            for i, error in enumerate(errors, 1):
                f.write(f"{i:3d}. 样本#{error['index']:4d}: "
                       f"真实={error['true_label']:15s} -> "
                       f"预测={error['pred_label']:15s} "
                       f"(置信度: {error['confidence']:.1f}%)\n")
    
    print(f"\n详细验证结果已保存到: {result_path}")
    
    print("\n" + "="*80)
    print("验证完成!")
    print("="*80)
    
    return {
        'accuracy': accuracy,
        'class_accuracies': class_accuracies,
        'errors': errors,
        'confusion_matrix': cm
    }


if __name__ == "__main__":
    # 设置GPU内存增长（如果使用GPU）
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # 验证模型
    print("\n" + "="*80)
    print("自动验证train_sample数据集")
    print("="*80)
    
    # 查找可用的模型
    model_name = f"cnn_sample_best.h5"
    model_path = os.path.join(MODEL_SAVE_DIR, model_name)
    
    if not os.path.exists(model_path):
        print(f"\n警告: 找不到模型 {model_path}")
        print("请先运行 train.py 训练模型!")
        
        # 尝试查找其他可用模型
        available_models = [f for f in os.listdir(MODEL_SAVE_DIR) if f.endswith('.h5')]
        if available_models:
            print(f"\n可用的模型:")
            for i, m in enumerate(available_models, 1):
                print(f"{i}. {m}")
            
            # 使用第一个可用模型
            model_path = os.path.join(MODEL_SAVE_DIR, available_models[0])
            print(f"\n使用模型: {available_models[0]}")
    
    # 执行验证
    results = validate_model(model_path, TRAIN_SAMPLE_DIR)
    
    print(f"\n{'='*80}")
    print(f"验证完成! 整体准确率: {results['accuracy']*100:.2f}%")
    print(f"{'='*80}")
