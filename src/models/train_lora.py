"""
CodeG - LoRA 微调训练脚本
使用 PEFT 库对 CodeBERT 进行 LoRA 微调
"""

import json
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset


def load_data(data_file):
    """加载数据"""
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def prepare_dataset(data, tokenizer, max_length=512):
    """准备数据集"""
    texts = [item['buggy_code'] for item in data]
    labels = [item['has_bug'] for item in data]

    # Tokenize
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=False,  # 动态 padding
        max_length=max_length
    )

    # 添加 labels
    encodings['labels'] = labels

    # 转换为 Dataset 格式
    dataset = Dataset.from_dict(encodings)
    return dataset


def main():
    # 配置
    model_name = "microsoft/codebert-base"
    data_dir = Path("/codeG/data/defects4j/processed")
    output_dir = "/codeG/models/lora-bug-detection"

    # 加载 tokenizer 和模型
    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2  # 二分类：有bug/无bug
    )

    # LoRA 配置
    print("配置 LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,  # 低秩维度
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["query", "key", "value", "dense"]  # CodeBERT 的注意力层
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 打印可训练参数

    # 加载数据
    print("加载数据...")
    train_data = load_data(data_dir / "train.json")
    val_data = load_data(data_dir / "val.json")

    train_dataset = prepare_dataset(train_data, tokenizer)
    val_dataset = prepare_dataset(val_data, tokenizer)

    # 数据整理器（动态 padding）
    data_collator = DataCollatorWithPadding(tokenizer)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),  # 混合精度训练
        report_to="none"  # 不上报到 wandb 等
    )

    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 训练
    print("开始训练...")
    trainer.train()

    # 保存模型
    print(f"保存模型到 {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("训练完成！")


if __name__ == '__main__':
    main()