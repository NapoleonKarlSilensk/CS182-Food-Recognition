# -*- coding: utf-8 -*-
"""
Administer experiment logging
"""
import os
import json
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

class Exp_Logger:
    """Experiment logger that automatically saves all experiment data and visualizations"""
    
    def __init__(self, exp_name, base_dir='experiments'):
        """
        Initialize the experiment logger
        
        Args:
            exp_name: Experiment name, e.g., 'baseline_cnn', 'yamnet_transfer'
            base_dir: Base directory for experiments
        """
        self.exp_name = exp_name
        self.base_dir = base_dir
        
        # Create experiment folder: experiments/baseline_cnn_20251220_030000/
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_id = self._get_next_exp_id()
        self.exp_folder = os.path.join(
            base_dir, 
            f'EXP{exp_id:03d}_{exp_name}_{timestamp}'
        )
        
        # Create subfolders
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
        
        # Experiment metadata
        self.metadata = {
            'experiment_id': f'EXP{exp_id:03d}',
            'exp_name': exp_name,
            'timestamp': timestamp,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        print(f"\n{'='*80}")
        print(f"Experiment {self.metadata['experiment_id']} initialized")
        print(f"Experiment name: {exp_name}")
        print(f"Experiment folder: {self.exp_folder}")
        print(f"{'='*80}\n")
    
    def _get_next_exp_id(self):
        """Get the next experiment ID"""
        if not os.path.exists(self.base_dir):
            return 1
        
        existing = [d for d in os.listdir(self.base_dir) if d.startswith('EXP')]
        if not existing:
            return 1
        
        exp_ids = [int(d.split('_')[0][3:]) for d in existing]
        return max(exp_ids) + 1
    
    def log_config(self, config_dict):
        """Log configuration parameters"""
        self.metadata['config'] = config_dict
        config_path = os.path.join(self.folders['logs'], 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"Configuration saved: {config_path}")
    
    def log_model(self, model):
        """Save model architecture summary"""
        summary_path = os.path.join(self.folders['logs'], 'model_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        print(f"Model architecture saved: {summary_path}")
    
    def log_history(self, history, save_plot=True):
        """Save training history"""
        # Save raw data
        history_path = os.path.join(self.folders['logs'], 'training_history.json')
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        print(f"Training history saved: {history_path}")
        
        # Plot training curves
        if save_plot:
            self._plot_curves(history_dict)
    
    def _plot_curves(self, history):
        """Plot training curves (accuracy and loss)"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy curve
        epochs = range(1, len(history['accuracy']) + 1)
        ax1.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        ax1.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Loss curve
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
        print(f"Training curves saved: {plot_path}")
    
    def log_metrics(self, metrics_dict, filename='metrics.json'):
        """Save performance metrics"""
        # Convert numpy types to native Python types
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_metrics = convert_to_serializable(metrics_dict)
        self.metadata['metrics'] = serializable_metrics
        metrics_path = os.path.join(self.folders['metrics'], filename)
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
        print(f"Performance metrics saved: {metrics_path}")
    
    def log_cm(self, cm, class_names, normalize=False):
        """Save confusion matrix"""
        import seaborn as sns
        
        # Save raw data
        cm_path = os.path.join(self.folders['metrics'], 'confusion_matrix.npy')
        np.save(cm_path, cm)
        
        # Plot confusion matrix
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
        print(f"Confusion matrix saved: {plot_path}")
    
    def log_report(self, report_text, report_dict=None):
        """Save classification report"""
        # Save text report
        text_path = os.path.join(self.folders['metrics'], 'classification_report.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"Classification report saved: {text_path}")
        
        # Save JSON format
        if report_dict:
            json_path = os.path.join(self.folders['metrics'], 'classification_report.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, indent=2)
    
    def save_model(self, model, model_name='best_model.h5'):
        """Save model"""
        model_path = os.path.join(self.folders['models'], model_name)
        model.save(model_path)
        print(f"Model saved: {model_path}")
        return model_path
    
    def save_predictions(self, predictions, filename='predictions.csv'):
        """Save predictions"""
        pred_path = os.path.join(self.folders['predictions'], filename)
        predictions.to_csv(pred_path, index=False)
        print(f"Predictions saved: {pred_path}")
    
    def finalize(self, final_metrics=None):
        """Complete experiment and generate summary report"""
        if final_metrics:
            self.metadata['final_metrics'] = final_metrics
        
        # Generate README
        readme_path = os.path.join(self.folders['root'], 'README.md')
        self._generate_readme(readme_path)
        
        # Save complete metadata
        metadata_path = os.path.join(self.folders['root'], 'experiment_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print(f"Experiment {self.metadata['experiment_id']} completed")
        print(f"All results saved to: {self.exp_folder}")
        print(f"{'='*80}\n")
    
    def _generate_readme(self, readme_path):
        """Generate experiment README"""
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"# {self.metadata['experiment_id']}: {self.metadata['exp_name']}\n\n")
            f.write(f"**Experiment Date**: {self.metadata['date']}\n\n")
            
            if 'config' in self.metadata:
                f.write("## Configuration\n\n")
                f.write("```json\n")
                f.write(json.dumps(self.metadata['config'], indent=2, ensure_ascii=False))
                f.write("\n```\n\n")
            
            if 'metrics' in self.metadata:
                f.write("## Performance Metrics\n\n")
                for key, value in self.metadata['metrics'].items():
                    f.write(f"- **{key}**: {value}\n")
                f.write("\n")
            
            f.write("## File Structure\n\n")
            f.write("```\n")
            f.write(f"{os.path.basename(self.exp_folder)}/\n")
            f.write("├── README.md                    # This file\n")
            f.write("├── experiment_metadata.json     # Complete metadata\n")
            f.write("├── plots/\n")
            f.write("│   ├── training_curves.png      # Training curves\n")
            f.write("│   └── confusion_matrix.png     # Confusion matrix\n")
            f.write("├── models/\n")
            f.write("│   └── best_model.h5            # Best model\n")
            f.write("├── logs/\n")
            f.write("│   ├── config.json              # Configuration\n")
            f.write("│   ├── model_summary.txt        # Model architecture\n")
            f.write("│   └── training_history.json    # Training history\n")
            f.write("├── metrics/\n")
            f.write("│   ├── metrics.json             # Performance metrics\n")
            f.write("│   ├── confusion_matrix.npy     # Confusion matrix data\n")
            f.write("│   ├── classification_report.txt\n")
            f.write("│   └── classification_report.json\n")
            f.write("└── predictions/\n")
            f.write("    └── predictions.csv          # Predictions\n")
            f.write("```\n")
        
        print(f"Experiment README generated: {readme_path}")


def create_experiments_index():
    """Create experiments index file"""
    index_path = 'experiments/INDEX.md'
    
    if not os.path.exists('experiments'):
        os.makedirs('experiments')
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write("# Experiments Index\n\n")
            f.write("This file records summary information for all experiments.\n\n")
            f.write("| Experiment ID | Experiment Name | Date | Validation Accuracy | Notes |\n")
            f.write("|---------------|-----------------|------|---------------------|-------|\n")
        print(f"✓ Experiments index created: {index_path}")
        return
    
    # Update index
    experiments = sorted([d for d in os.listdir('experiments') if d.startswith('EXP')])
    
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write("# Experiments Index\n\n")
        f.write("This file records summary information for all experiments.\n\n")
        f.write("| Experiment ID | Experiment Name | Date | Validation Accuracy | Training Time | Notes |\n")
        f.write("|---------------|-----------------|------|---------------------|---------------|-------|\n")
        
        for exp_dir in experiments:
            metadata_path = os.path.join('experiments', exp_dir, 'experiment_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as mf:
                    metadata = json.load(mf)
                    
                exp_id = metadata.get('experiment_id', 'N/A')
                exp_name = metadata.get('exp_name', 'N/A')
                date = metadata.get('date', 'N/A')
                
                metrics = metadata.get('metrics', {})
                val_acc = metrics.get('validation_accuracy', 'N/A')
                if val_acc != 'N/A':
                    val_acc = f"{val_acc*100:.2f}%" if isinstance(val_acc, float) else val_acc
                
                train_time = metrics.get('training_time', 'N/A')
                note = metadata.get('note', '-')
                
                f.write(f"| [{exp_id}]({exp_dir}/README.md) | {exp_name} | {date} | {val_acc} | {train_time} | {note} |\n")


if __name__ == '__main__':
    # Test example
    logger = Exp_Logger('test_experiment')
    
    # Simulate configuration
    config = {
        'model': 'CNN',
        'epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001
    }
    logger.log_config(config)
    
    # Simulate metrics
    metrics = {
        'validation_accuracy': 0.75,
        'validation_loss': 1.2,
        'training_time': '30 minutes'
    }
    logger.log_metrics(metrics)
    
    logger.finalize(metrics)
    create_experiments_index()