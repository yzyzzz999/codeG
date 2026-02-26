"""
CodeG - 模型推理脚本
预测代码是否有 Bug
"""

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel


class BugDetector:
    """Bug 检测器"""

    def __init__(self, model_path: str, base_model_name: str = "microsoft/codebert-base"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载 tokenizer 和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=2
        )
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, code: str, max_length: int = 512):
        """
        预测代码是否有 Bug

        Returns:
            dict: {
                'has_bug': bool,
                'confidence': float,  # 置信度
                'bug_probability': float  # 有bug的概率
            }
        """
        # Tokenize
        encoding = self.tokenizer(
            code,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # 预测
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        bug_prob = probs[0][1].item()  # 有bug的概率
        no_bug_prob = probs[0][0].item()

        has_bug = bug_prob > 0.5
        confidence = max(bug_prob, no_bug_prob)

        return {
            'has_bug': has_bug,
            'confidence': confidence,
            'bug_probability': bug_prob
        }


def main():
    # 加载模型
    model_path = "/codeG/models/lora-bug-detection"
    detector = BugDetector(model_path)

    # 测试代码
    test_code = """
    public int divide(int a, int b) {
        return a / b;  // 可能的除零错误
    }
    """

    # 预测
    result = detector.predict(test_code)

    print("预测结果:")
    print(f"  有 Bug: {result['has_bug']}")
    print(f"  置信度: {result['confidence']:.4f}")
    print(f"  Bug 概率: {result['bug_probability']:.4f}")

    # 批量预测示例
    print("\n批量预测测试集...")
    with open("/codeG/data/defects4j/processed/test.json", 'r') as f:
        test_data = json.load(f)

    correct = 0
    for item in test_data[:5]:  # 只测前5个
        result = detector.predict(item['buggy_code'])
        pred = 1 if result['has_bug'] else 0
        actual = item['has_bug']

        status = "✓" if pred == actual else "✗"
        print(f"{status} 预测: {pred}, 实际: {actual}, 置信度: {result['confidence']:.4f}")

        if pred == actual:
            correct += 1

    print(f"\n准确率: {correct}/5 = {correct / 5 * 100:.1f}%")


if __name__ == '__main__':
    main()