"""
数据划分脚本
将 defects4j_pairs.json 划分为 train/val/test
"""
import json
import random
from pathlib import Path
from typing import List, Dict

def split_dataset(
    input_file: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    """划分数据集"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须等于1"
    
    # 加载数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总样本数: {len(data)}")
    
    # 按项目分组，确保每个项目在各集合中都有代表
    projects = {}
    for item in data:
        proj = item['bug_id'].split('_')[0]
        if proj not in projects:
            projects[proj] = []
        projects[proj].append(item)
    
    print(f"\n项目分布:")
    for proj, items in sorted(projects.items()):
        print(f"  {proj}: {len(items)} 个样本")
    
    # 划分每个项目的数据
    random.seed(seed)
    train_data = []
    val_data = []
    test_data = []
    
    for proj, items in projects.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_data.extend(items[:n_train])
        val_data.extend(items[n_train:n_train + n_val])
        test_data.extend(items[n_train + n_val:])
    
    # 转换为模型需要的格式
    def convert_format(data: List[Dict]) -> List[Dict]:
        """转换为模型输入格式"""
        result = []
        for item in data:
            # Buggy 样本 (label=1)
            result.append({
                'code': item['buggy_code'],
                'label': 1,
                'bug_id': item['bug_id']
            })
            # Fixed 样本 (label=0)
            result.append({
                'code': item['fixed_code'],
                'label': 0,
                'bug_id': item['bug_id']
            })
        return result
    
    train_converted = convert_format(train_data)
    val_converted = convert_format(val_data)
    test_converted = convert_format(test_data)
    
    # 保存
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'train.json', 'w', encoding='utf-8') as f:
        json.dump(train_converted, f, indent=2, ensure_ascii=False)
    
    with open(output_path / 'val.json', 'w', encoding='utf-8') as f:
        json.dump(val_converted, f, indent=2, ensure_ascii=False)
    
    with open(output_path / 'test.json', 'w', encoding='utf-8') as f:
        json.dump(test_converted, f, indent=2, ensure_ascii=False)
    
    print(f"\n划分结果:")
    print(f"  Train: {len(train_data)} 对 ({len(train_converted)} 条) - {len(train_data)/len(data)*100:.1f}%")
    print(f"  Val:   {len(val_data)} 对 ({len(val_converted)} 条) - {len(val_data)/len(data)*100:.1f}%")
    print(f"  Test:  {len(test_data)} 对 ({len(test_converted)} 条) - {len(test_data)/len(data)*100:.1f}%")
    print(f"\n保存到: {output_path}")

if __name__ == '__main__':
    split_dataset(
        input_file="/codeG/data/defects4j/processed/defects4j_pairs.json",
        output_dir="/codeG/data/defects4j/split",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
