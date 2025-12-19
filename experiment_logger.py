# -*- coding: utf-8 -*-
"""
实验管理系统 - 自动记录和存档每次实验的结果
"""
import os
import json
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

class ExperimentLogger:
    """实验日志记录器，自动保存所有实验数据和可视化"""
    
    def __init__(self, experiment_name, base_dir='experiments'):
        """
        初始化实验记录器
        
        Args:
            experiment_name: 实验名称，如 'baseline_cnn', 'yamnet_transfer'
            base_dir: 实验根目录
        """
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        
        # 创建实验文件夹：experiments/EXP001_baseline_cnn_20251220_030000/
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_id = self._get_next_exp_id()
        self.exp_folder = os.path.join(
            base_dir, 
            f'EXP{exp_id:03d}_{experiment_name}_{timestamp}'
        )
        
        # 创建子文件夹
        self.folders = {
            'root': self.exp_folder,
            'plots': os.path.join(self.exp_folder, 'plots'),
            'models': os.path.join(self.exp_folder, 'models'),
            'logs': os.path.join(self.exp_folder, 'logs'),
            'predictions': os.path.join(self.exp_folder, 'predictions'),
            'metrics': os.path.join(self.exp_folder, 'metrics'),
        }
        
        for folder in self.folders.values():
            os.makedirs(folder, exist_ok=True)
        
        # 实验元数据
        self.metadata = {
            'experiment_id': f'EXP{exp_id:03d}',
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        print(f"\n{'='*80}")
        print(f"实验 {self.metadata['experiment_id']} 已初始化")
        print(f"实验名称: {experiment_name}")
        print(f"实验文件夹: {self.exp_folder}")
        print(f"{'='*80}\n")
    
    def _get_next_exp_id(self):
        """获取下一个实验编号"""
        if not os.path.exists(self.base_dir):
            return 1
        
        existing = [d for d in os.listdir(self.base_dir) if d.startswith('EXP')]
        if not existing:
            return 1
        
        exp_ids = [int(d.split('_')[0][3:]) for d in existing]
        return max(exp_ids) + 1
    
    def log_config(self, config_dict):
        """记录配置参数"""
        self.metadata['config'] = config_dict
        config_path = os.path.join(self.folders['logs'], 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"✓ 配置已保存: {config_path}")
    
    def log_model_summary(self, model):
        """保存模型架构摘要"""
        summary_path = os.path.join(self.folders['logs'], 'model_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        print(f"✓ 模型架构已保存: {summary_path}")
    
    def log_training_history(self, history, save_plot=True):
        """保存训练历史"""
        # 保存原始数据
        history_path = os.path.join(self.folders['logs'], 'training_history.json')
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        print(f"✓ 训练历史已保存: {history_path}")
        
        # 绘制训练曲线
        if save_plot:
            self._plot_training_curves(history_dict)
    
    def _plot_training_curves(self, history):
        """绘制训练曲线（准确率和损失）"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 准确率曲线
        epochs = range(1, len(history['accuracy']) + 1)
        ax1.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        ax1.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 损失曲线
        ax2.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
        ax2.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.folders['plots'], 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 训练曲线已保存: {plot_path}")
    
    def log_metrics(self, metrics_dict, filename='metrics.json'):
        """保存性能指标"""
        self.metadata['metrics'] = metrics_dict
        metrics_path = os.path.join(self.folders['metrics'], filename)
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
        print(f"✓ 性能指标已保存: {metrics_path}")
    
    def log_confusion_matrix(self, cm, class_names, normalize=False):
        """保存混淆矩阵"""
        import seaborn as sns
        
        # 保存原始数据
        cm_path = os.path.join(self.folders['metrics'], 'confusion_matrix.npy')
        np.save(cm_path, cm)
        
        # 绘制混淆矩阵
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(16, 14))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                    cmap='Blues', xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=13)
        plt.ylabel('True Label', fontsize=13)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plot_path = os.path.join(self.folders['plots'], 'confusion_matrix.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 混淆矩阵已保存: {plot_path}")
    
    def log_classification_report(self, report_text, report_dict=None):
        """保存分类报告"""
        # 保存文本报告
        text_path = os.path.join(self.folders['metrics'], 'classification_report.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"✓ 分类报告已保存: {text_path}")
        
        # 保存JSON格式
        if report_dict:
            json_path = os.path.join(self.folders['metrics'], 'classification_report.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, indent=2)
    
    def save_model(self, model, model_name='best_model.h5'):
        """保存模型"""
        model_path = os.path.join(self.folders['models'], model_name)
        model.save(model_path)
        print(f"✓ 模型已保存: {model_path}")
        return model_path
    
    def save_predictions(self, predictions, filename='predictions.csv'):
        """保存预测结果"""
        pred_path = os.path.join(self.folders['predictions'], filename)
        predictions.to_csv(pred_path, index=False)
        print(f"✓ 预测结果已保存: {pred_path}")
    
    def finalize(self, final_metrics=None):
        """完成实验，生成总结报告"""
        if final_metrics:
            self.metadata['final_metrics'] = final_metrics
        
        # 生成README
        readme_path = os.path.join(self.folders['root'], 'README.md')
        self._generate_readme(readme_path)
        
        # 保存完整元数据
        metadata_path = os.path.join(self.folders['root'], 'experiment_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print(f"✓ 实验 {self.metadata['experiment_id']} 已完成")
        print(f"✓ 所有结果已保存到: {self.exp_folder}")
        print(f"{'='*80}\n")
    
    def _generate_readme(self, readme_path):
        """生成实验README"""
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"# {self.metadata['experiment_id']}: {self.metadata['experiment_name']}\n\n")
            f.write(f"**实验日期**: {self.metadata['date']}\n\n")
            
            if 'config' in self.metadata:
                f.write("## 配置参数\n\n")
                f.write("```json\n")
                f.write(json.dumps(self.metadata['config'], indent=2, ensure_ascii=False))
                f.write("\n```\n\n")
            
            if 'metrics' in self.metadata:
                f.write("## 性能指标\n\n")
                for key, value in self.metadata['metrics'].items():
                    f.write(f"- **{key}**: {value}\n")
                f.write("\n")
            
            f.write("## 文件结构\n\n")
            f.write("```\n")
            f.write(f"{os.path.basename(self.exp_folder)}/\n")
            f.write("├── README.md                    # 本文件\n")
            f.write("├── experiment_metadata.json     # 完整元数据\n")
            f.write("├── plots/\n")
            f.write("│   ├── training_curves.png      # 训练曲线\n")
            f.write("│   └── confusion_matrix.png     # 混淆矩阵\n")
            f.write("├── models/\n")
            f.write("│   └── best_model.h5            # 最佳模型\n")
            f.write("├── logs/\n")
            f.write("│   ├── config.json              # 配置参数\n")
            f.write("│   ├── model_summary.txt        # 模型架构\n")
            f.write("│   └── training_history.json    # 训练历史\n")
            f.write("├── metrics/\n")
            f.write("│   ├── metrics.json             # 性能指标\n")
            f.write("│   ├── confusion_matrix.npy     # 混淆矩阵数据\n")
            f.write("│   ├── classification_report.txt\n")
            f.write("│   └── classification_report.json\n")
            f.write("└── predictions/\n")
            f.write("    └── predictions.csv          # 预测结果\n")
            f.write("```\n")
        
        print(f"✓ 实验README已生成: {readme_path}")


def create_experiments_index():
    """创建实验索引文件"""
    index_path = 'experiments/INDEX.md'
    
    if not os.path.exists('experiments'):
        os.makedirs('experiments')
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write("# 实验索引\n\n")
            f.write("本文件记录所有实验的概要信息。\n\n")
            f.write("| 实验ID | 实验名称 | 日期 | 验证准确率 | 说明 |\n")
            f.write("|--------|----------|------|-----------|------|\n")
        print(f"✓ 实验索引已创建: {index_path}")
        return
    
    # 更新索引
    experiments = sorted([d for d in os.listdir('experiments') if d.startswith('EXP')])
    
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write("# 实验索引\n\n")
        f.write("本文件记录所有实验的概要信息。\n\n")
        f.write("| 实验ID | 实验名称 | 日期 | 验证准确率 | 训练时长 | 说明 |\n")
        f.write("|--------|----------|------|-----------|---------|------|\n")
        
        for exp_dir in experiments:
            metadata_path = os.path.join('experiments', exp_dir, 'experiment_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as mf:
                    metadata = json.load(mf)
                    
                exp_id = metadata.get('experiment_id', 'N/A')
                exp_name = metadata.get('experiment_name', 'N/A')
                date = metadata.get('date', 'N/A')
                
                metrics = metadata.get('metrics', {})
                val_acc = metrics.get('validation_accuracy', 'N/A')
                if val_acc != 'N/A':
                    val_acc = f"{val_acc*100:.2f}%" if isinstance(val_acc, float) else val_acc
                
                train_time = metrics.get('training_time', 'N/A')
                note = metadata.get('note', '-')
                
                f.write(f"| [{exp_id}]({exp_dir}/README.md) | {exp_name} | {date} | {val_acc} | {train_time} | {note} |\n")


if __name__ == '__main__':
    # 测试示例
    logger = ExperimentLogger('test_experiment')
    
    # 模拟配置
    config = {
        'model': 'CNN',
        'epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001
    }
    logger.log_config(config)
    
    # 模拟指标
    metrics = {
        'validation_accuracy': 0.75,
        'validation_loss': 1.2,
        'training_time': '30 minutes'
    }
    logger.log_metrics(metrics)
    
    logger.finalize(metrics)
    create_experiments_index()
    
    print(f"\n实验文件夹已创建，可查看: {logger.exp_folder}")
