"""
CodeG - LoRA 微调训练脚本（实验版）
支持参数调整和实验记录
"""

import json
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


class BugDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=512):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['code'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def eval_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    tn, fp, fn, tp = conf_matrix.ravel() if conf_matrix.size == 4 else (0, 0, 0, 0)
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }


def train_with_config(config, exp_id=None):
    """使用指定配置训练模型"""
    
    # 实验ID
    if exp_id is None:
        exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*60}")
    print(f"实验 ID: {exp_id}")
    print(f"配置: {json.dumps(config, indent=2)}")
    print(f"{'='*60}\n")
    
    # 路径配置
    model_name = "microsoft/codebert-base"
    output_dir = Path(f"/codeG/models/exp_{exp_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_split_dir = Path("/codeG/data/defects4j/split")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        local_files_only=True
    )
    
    # LoRA 配置
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        target_modules=["query", "key", "value", "dense"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)
    
    # 加载数据
    print("加载数据...")
    train_dataset = BugDataset(data_split_dir / "train.json", tokenizer)
    val_dataset = BugDataset(data_split_dir / "val.json", tokenizer)
    test_dataset = BugDataset(data_split_dir / "test.json", tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    # 训练记录
    history = {
        'exp_id': exp_id,
        'config': config,
        'epochs': []
    }
    
    best_val_acc = 0
    
    # 训练循环
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = eval_epoch(model, val_loader, device)
        
        epoch_record = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'val_tn': val_metrics['tn'],
            'val_fp': val_metrics['fp'],
            'val_fn': val_metrics['fn'],
            'val_tp': val_metrics['tp']
        }
        history['epochs'].append(epoch_record)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val  Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"P: {val_metrics['precision']:.4f}, R: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"Val  Confusion: TN={val_metrics['tn']}, FP={val_metrics['fp']}, "
              f"FN={val_metrics['fn']}, TP={val_metrics['tp']}")
        
        # 保存最佳模型
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"✓ 保存最佳模型到 {output_dir}")
    
    # 最终测试集评估
    print(f"\n{'='*60}")
    print("测试集评估")
    print(f"{'='*60}")
    test_metrics = eval_epoch(model, test_loader, device)
    
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Acc: {test_metrics['accuracy']:.4f}, "
          f"P: {test_metrics['precision']:.4f}, R: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")
    print(f"Test Confusion: TN={test_metrics['tn']}, FP={test_metrics['fp']}, "
          f"FN={test_metrics['fn']}, TP={test_metrics['tp']}")
    
    history['test'] = {
        'loss': test_metrics['loss'],
        'accuracy': test_metrics['accuracy'],
        'precision': test_metrics['precision'],
        'recall': test_metrics['recall'],
        'f1': test_metrics['f1'],
        'tn': test_metrics['tn'],
        'fp': test_metrics['fp'],
        'fn': test_metrics['fn'],
        'tp': test_metrics['tp']
    }
    
    # 保存实验记录
    exp_file = output_dir / "experiment.json"
    with open(exp_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"\n实验记录已保存: {exp_file}")
    
    return history


def main():
    # 实验配置 - 降低 LoRA rank 和增加 dropout 来减少过拟合
    config = {
        'lora_r': 8,           # 从 16 降到 8
        'lora_alpha': 16,      # 从 32 降到 16
        'lora_dropout': 0.2,   # 从 0.1 增加到 0.2
        'lr': 1e-5,            # 从 2e-5 降到 1e-5
        'batch_size': 4,       # 从 2 增加到 4
        'epochs': 15,          # 从 10 增加到 15
        'weight_decay': 0.05   # 新增正则化
    }
    
    train_with_config(config)


if __name__ == '__main__':
    main()
