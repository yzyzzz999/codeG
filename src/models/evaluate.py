"""
CodeG - 模型评估脚本
评估 Bug 检测模型的性能
"""

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm


def load_model(model_path, base_model_name="microsoft/codebert-base"):
    """加载训练好的模型"""
    print(f"加载模型: {model_path}")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 加载基础模型 + LoRA 权重
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    return model, tokenizer


def evaluate(model, tokenizer, data_file, device, max_length=512):
    """评估模型"""
    # 加载数据
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    predictions = []
    labels = []

    print(f"评估 {len(data)} 个样本...")

    with torch.no_grad():
        for item in tqdm(data):
            # Tokenize
            encoding = tokenizer(
                item['code'],
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            # 预测
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=-1).item()

            predictions.append(pred)
            labels.append(item['label'])

    # 计算指标
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    conf_matrix = confusion_matrix(labels, predictions)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix.tolist(),
        'predictions': predictions,
        'labels': labels
    }


def main():
    # 配置
    model_path = "/codeG/models/lora-bug-detection"
    data_file = "/codeG/data/defects4j/split/test.json"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    model, tokenizer = load_model(model_path)
    model.to(device)

    # 评估
    results = evaluate(model, tokenizer, data_file, device)

    # 打印结果
    print("\n" + "=" * 50)
    print("评估结果")
    print("=" * 50)
    print(f"准确率 (Accuracy): {results['accuracy']:.4f}")
    print(f"精确率 (Precision): {results['precision']:.4f}")
    print(f"召回率 (Recall): {results['recall']:.4f}")
    print(f"F1 分数: {results['f1']:.4f}")
    print("\n混淆矩阵:")
    if len(results['confusion_matrix']) >= 2:
        print(f"TN={results['confusion_matrix'][0][0]}, FP={results['confusion_matrix'][0][1]}")
        print(f"FN={results['confusion_matrix'][1][0]}, TP={results['confusion_matrix'][1][1]}")
    else:
        print(f"混淆矩阵: {results['confusion_matrix']}")

    # 保存结果
    output_file = Path(model_path) / "evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_file}")


if __name__ == '__main__':
    main()