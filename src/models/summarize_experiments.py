"""
实验结果汇总脚本
生成实验结果表格
"""
import json
import csv
from pathlib import Path
from datetime import datetime


def collect_experiments(models_dir="/codeG/models"):
    """收集所有实验结果"""
    models_path = Path(models_dir)
    experiments = []
    
    for exp_dir in models_path.glob("exp_*"):
        exp_file = exp_dir / "experiment.json"
        if exp_file.exists():
            with open(exp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取关键信息
            exp_record = {
                'exp_id': data['exp_id'],
                'timestamp': data['exp_id'],  # exp_id 就是时间戳
                'lora_r': data['config']['lora_r'],
                'lora_alpha': data['config']['lora_alpha'],
                'lora_dropout': data['config']['lora_dropout'],
                'lr': data['config']['lr'],
                'batch_size': data['config']['batch_size'],
                'epochs': data['config']['epochs'],
                'weight_decay': data['config'].get('weight_decay', 0),
                # 训练集最佳结果 (取最后一轮)
                'train_loss': data['epochs'][-1]['train_loss'] if data['epochs'] else None,
                # 验证集最佳结果
                'best_val_epoch': max(range(len(data['epochs'])), 
                                      key=lambda i: data['epochs'][i]['val_accuracy']) + 1 if data['epochs'] else None,
                'val_accuracy': max([e['val_accuracy'] for e in data['epochs']]) if data['epochs'] else None,
                'val_precision': data['epochs'][max(range(len(data['epochs'])), 
                                                     key=lambda i: data['epochs'][i]['val_accuracy'])]['val_precision'] if data['epochs'] else None,
                'val_recall': data['epochs'][max(range(len(data['epochs'])), 
                                                 key=lambda i: data['epochs'][i]['val_accuracy'])]['val_recall'] if data['epochs'] else None,
                'val_f1': data['epochs'][max(range(len(data['epochs'])), 
                                             key=lambda i: data['epochs'][i]['val_accuracy'])]['val_f1'] if data['epochs'] else None,
                # 测试集结果
                'test_accuracy': data['test']['accuracy'],
                'test_precision': data['test']['precision'],
                'test_recall': data['test']['recall'],
                'test_f1': data['test']['f1'],
                'test_tn': data['test']['tn'],
                'test_fp': data['test']['fp'],
                'test_fn': data['test']['fn'],
                'test_tp': data['test']['tp']
            }
            experiments.append(exp_record)
    
    # 按时间排序
    experiments.sort(key=lambda x: x['timestamp'])
    return experiments


def generate_csv(experiments, output_file="/codeG/experiments_summary.csv"):
    """生成 CSV 表格"""
    if not experiments:
        print("没有找到实验记录")
        return
    
    fieldnames = [
        'exp_id', 'timestamp',
        'lora_r', 'lora_alpha', 'lora_dropout', 'lr', 'batch_size', 'epochs', 'weight_decay',
        'train_loss', 'best_val_epoch', 
        'val_accuracy', 'val_precision', 'val_recall', 'val_f1',
        'test_accuracy', 'test_precision', 'test_recall', 'test_f1',
        'test_tn', 'test_fp', 'test_fn', 'test_tp'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(experiments)
    
    print(f"实验汇总表已保存: {output_file}")
    print(f"共 {len(experiments)} 条实验记录")


def generate_markdown(experiments, output_file="/codeG/experiments_summary.md"):
    """生成 Markdown 表格"""
    if not experiments:
        print("没有找到实验记录")
        return
    
    lines = [
        "# CodeG 实验结果汇总",
        "",
        f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 实验配置与结果",
        "",
        "| 实验ID | LoRA r | alpha | dropout | LR | batch | epochs | wd |",
        "|--------|--------|-------|---------|-------|-------|--------|-----|",
    ]
    
    for exp in experiments:
        lines.append(
            f"| {exp['exp_id']} | {exp['lora_r']} | {exp['lora_alpha']} | "
            f"{exp['lora_dropout']} | {exp['lr']:.0e} | {exp['batch_size']} | "
            f"{exp['epochs']} | {exp['weight_decay']} |"
        )
    
    lines.extend([
        "",
        "## 验证集结果",
        "",
        "| 实验ID | Best Epoch | Acc | Precision | Recall | F1 |",
        "|--------|------------|-----|-----------|--------|-----|",
    ])
    
    for exp in experiments:
        lines.append(
            f"| {exp['exp_id']} | {exp['best_val_epoch']} | "
            f"{exp['val_accuracy']:.4f} | {exp['val_precision']:.4f} | "
            f"{exp['val_recall']:.4f} | {exp['val_f1']:.4f} |"
        )
    
    lines.extend([
        "",
        "## 测试集结果",
        "",
        "| 实验ID | Acc | Precision | Recall | F1 | TN | FP | FN | TP |",
        "|--------|-----|-----------|--------|-----|----|----|----|----|",
    ])
    
    for exp in experiments:
        lines.append(
            f"| {exp['exp_id']} | {exp['test_accuracy']:.4f} | "
            f"{exp['test_precision']:.4f} | {exp['test_recall']:.4f} | "
            f"{exp['test_f1']:.4f} | {exp['test_tn']} | {exp['test_fp']} | "
            f"{exp['test_fn']} | {exp['test_tp']} |"
        )
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"Markdown 汇总表已保存: {output_file}")


def main():
    experiments = collect_experiments()
    
    if experiments:
        generate_csv(experiments)
        generate_markdown(experiments)
        
        # 打印最新实验结果
        print("\n" + "="*80)
        print("最新实验结果:")
        print("="*80)
        latest = experiments[-1]
        print(f"实验ID: {latest['exp_id']}")
        print(f"配置: r={latest['lora_r']}, alpha={latest['lora_alpha']}, "
              f"dropout={latest['lora_dropout']}, lr={latest['lr']:.0e}")
        print(f"验证集: Acc={latest['val_accuracy']:.4f}, F1={latest['val_f1']:.4f}")
        print(f"测试集: Acc={latest['test_accuracy']:.4f}, F1={latest['test_f1']:.4f}")
    else:
        print("暂无实验记录")


if __name__ == '__main__':
    main()
