"""
CodeG - 数据集
加载 JSON 数据（train/val/test）
用 CodeBERT tokenizer 编码代码
返回模型可用的格式
"""
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class BugDetectionDataset(Dataset):
    """数据集"""

    def __init__(self,
                 data_path: str,
                 model_name: str = "microsoft/codebert-base",
                 max_len: int = 512,
                 device: str = None):
        self.data_path = Path(data_path)
        self.max_len = max_len
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.samples = self._load_data()


    def _load_data(self) -> list[dict]:
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 验证数据格式
        if not data:
            raise ValueError(f"Empty dataset: {self.data_path}")
        required_keys = {'code', 'label'}
        for i, sample in enumerate(data[:5]):  # 检查前5条
            if not required_keys.issubset(sample.keys()):
                raise KeyError(f"Sample {i} missing keys: {required_keys - sample.keys()}")
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        code = sample['code']
        label = sample['label']

        inputs = self.tokenizer(
            code,
            padding=False,
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(0).to(self.device),
            'attention_mask': inputs['attention_mask'].squeeze(0).to(self.device),
            'labels': torch.tensor(label, dtype=torch.long, device=self.device)
        }

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        将多个样本拼接成一个 batch
        """
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        padding_value = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else 0

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=padding_value
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_masks,
            batch_first=True,
            padding_value=padding_value
        )
        labels = torch.stack(labels)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def create_dataloaders(
        train_path: str,
        val_path: str,
        test_path: str,
        batch_size: int = 8,
        model_name: str = "microsoft/codebert-base",
        max_len: int = 512,
        num_workers: int = 0
) -> Dict[str, DataLoader]:
    train_dataset = BugDetectionDataset(train_path, model_name, max_len)
    val_dataset = BugDetectionDataset(val_path, model_name, max_len)
    test_dataset = BugDetectionDataset(test_path, model_name, max_len)

    datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }

    dataloaders = {}
    for name, dataset in datasets.items():
        dataloaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(name == 'train'),
            collate_fn=collate_fn,
            num_workers=num_workers
        )

    return dataloaders


if __name__ == '__main__':
    # 测试代码
    data_dir = Path(__file__).parent.parent.parent / "data" / "defects4j" / "mock"

    # 测试 Dataset
    print("=== 测试 Dataset ===")
    dataset = BugDetectionDataset(data_dir / "train.json")
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Label: {sample['labels']}")

    # 测试 DataLoader
    print("\n=== 测试 DataLoader ===")
    dataloaders = create_dataloaders(
        train_path=data_dir / "train.json",
        val_path=data_dir / "val.json",
        test_path=data_dir / "test.json",
        batch_size=4
    )
    batch = next(iter(dataloaders['train']))
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch labels: {batch['labels']}")

    print("\n测试完成！")