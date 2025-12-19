# -*- coding: utf-8 -*-
"""
归档历史实验结果
"""
import os
import shutil
import json
from datetime import datetime
from experiment_logger import ExperimentLogger, create_experiments_index

def archive_baseline_experiments():
    """归档之前的Baseline实验"""
    
    print("="*80)
    print("归档历史实验结果...")
    print("="*80)
    
    # 实验1: Baseline CNN (train_sample) - 36%
    print("\n[1/3] 归档 EXP002: Baseline CNN (train_sample)...")
    exp002 = ExperimentLogger('baseline_cnn_sample')
    exp002.metadata['note'] = 'Simplified 3-layer CNN, train_sample dataset'
    
    config_002 = {
        'model_type': 'CNN',
        'architecture': '3-layer Simplified CNN',
        'conv_layers': 3,
        'filters': [32, 64, 128],
        'dense_units': 128,
        'dropout_rates': [0.3, 0.4, 0.5, 0.5],
        'l2_regularization': 0.01,
        'dataset': 'train_sample',
        'train_samples': 800,
        'val_samples': 200,
        'epochs': 200,
        'batch_size': 32,
        'learning_rate': 0.001,
        'optimizer': 'Adam',
        'callbacks': ['EarlyStopping(patience=20)', 'ReduceLROnPlateau(patience=8)']
    }
    exp002.log_config(config_002)
    
    metrics_002 = {
        'validation_accuracy': 0.36,
        'validation_loss': 2.1951,
        'training_time': '约20分钟',
        'best_epoch': 176,
        'total_params': 111764
    }
    exp002.log_metrics(metrics_002)
    
    # 复制训练曲线
    if os.path.exists('models/cnn_sample_training_history.png'):
        shutil.copy('models/cnn_sample_training_history.png', 
                   os.path.join(exp002.folders['plots'], 'training_curves.png'))
    
    # 复制混淆矩阵
    if os.path.exists('models/confusion_matrix_validation.png'):
        shutil.copy('models/confusion_matrix_validation.png',
                   os.path.join(exp002.folders['plots'], 'confusion_matrix.png'))
    
    # 复制验证报告
    if os.path.exists('models/validation_results.txt'):
        shutil.copy('models/validation_results.txt',
                   os.path.join(exp002.folders['metrics'], 'validation_results.txt'))
    
    # 复制模型
    if os.path.exists('models/cnn_sample_best.h5'):
        shutil.copy('models/cnn_sample_best.h5',
                   os.path.join(exp002.folders['models'], 'best_model.h5'))
    
    exp002.finalize(metrics_002)
    print("✓ EXP002 归档完成")
    
    # 实验2: Baseline CNN (full train) - 57.86%
    print("\n[2/3] 归档 EXP003: Baseline CNN (full train)...")
    exp003 = ExperimentLogger('baseline_cnn_full')
    exp003.metadata['note'] = 'Simplified 3-layer CNN, full train dataset - Current baseline'
    
    config_003 = {
        'model_type': 'CNN',
        'architecture': '3-layer Simplified CNN',
        'conv_layers': 3,
        'filters': [32, 64, 128],
        'dense_units': 128,
        'dropout_rates': [0.3, 0.4, 0.5, 0.5],
        'l2_regularization': 0.01,
        'dataset': 'train (full)',
        'train_samples': 5600,
        'val_samples': 1400,
        'total_samples': 7000,
        'num_classes': 20,
        'epochs': 200,
        'batch_size': 32,
        'learning_rate': 0.001,
        'optimizer': 'Adam',
        'callbacks': ['EarlyStopping(patience=20)', 'ReduceLROnPlateau(patience=8)', 'ModelCheckpoint'],
        'early_stopping_epoch': 199
    }
    exp003.log_config(config_003)
    
    metrics_003 = {
        'validation_accuracy': 0.5786,
        'validation_loss': 1.4183,
        'training_accuracy': 'N/A',
        'training_time': '85.44 minutes',
        'best_epoch': 199,
        'total_params': 111764,
        'trainable_params': 111764,
        'note': 'Current baseline - 从头训练的3层CNN'
    }
    exp003.log_metrics(metrics_003)
    
    # 复制文件
    if os.path.exists('models/cnn_full_training_history.png'):
        shutil.copy('models/cnn_full_training_history.png',
                   os.path.join(exp003.folders['plots'], 'training_curves.png'))
    
    if os.path.exists('models/cnn_full_best.h5'):
        shutil.copy('models/cnn_full_best.h5',
                   os.path.join(exp003.folders['models'], 'best_model.h5'))
    
    if os.path.exists('models/cnn_full_info.txt'):
        shutil.copy('models/cnn_full_info.txt',
                   os.path.join(exp003.folders['logs'], 'training_info.txt'))
    
    # 复制提交文件
    if os.path.exists('submissions/submission_test_a_20251220_025935.csv'):
        shutil.copy('submissions/submission_test_a_20251220_025935.csv',
                   os.path.join(exp003.folders['predictions'], 'test_a_predictions.csv'))
    
    if os.path.exists('submissions/submission_test_b_20251220_025935.csv'):
        shutil.copy('submissions/submission_test_b_20251220_025935.csv',
                   os.path.join(exp003.folders['predictions'], 'test_b_predictions.csv'))
    
    exp003.finalize(metrics_003)
    print("✓ EXP003 归档完成")
    
    # 清理测试实验
    print("\n[3/3] 清理测试实验...")
    test_exp = 'experiments/EXP001_test_experiment_20251220_032336'
    if os.path.exists(test_exp):
        shutil.rmtree(test_exp)
        print(f"✓ 已删除测试实验: {test_exp}")
    
    # 更新实验索引
    print("\n更新实验索引...")
    create_experiments_index()
    
    print("\n" + "="*80)
    print("✓ 所有历史实验已归档完成!")
    print("="*80)
    print("\n实验结果位于: experiments/")
    print("- EXP002: baseline_cnn_sample (36% accuracy)")
    print("- EXP003: baseline_cnn_full (57.86% accuracy) ← 当前baseline")
    print("\n下一个实验将从 EXP004 开始")


if __name__ == '__main__':
    archive_baseline_experiments()
