"""
Defects4J 模拟数据生成器
用于本地测试，无需下载真实数据集
"""

import json
import random
from pathlib import Path
from typing import List, Dict


# 模拟的 Java 方法模板
BUGGY_TEMPLATES = [
    '''public int {name}(int a, int b) {{
        // Bug: 应该返回 a + b，但返回了 a - b
        return a - b;
    }}''',
    '''public String {name}(String s) {{
        // Bug: 没有判空
        return s.toUpperCase();
    }}''',
    '''public void {name}(List<Integer> list) {{
        // Bug: 并发修改异常
        for (int i = 0; i < list.size(); i++) {{
            if (list.get(i) < 0) {{
                list.remove(i);
            }}
        }}
    }}''',
    '''public boolean {name}(String email) {{
        // Bug: 错误的正则
        return email.contains("@");
    }}''',
    '''public double {name}(double x) {{
        // Bug: 除以零
        return 1 / x;
    }}''',
]

FIXED_TEMPLATES = [
    '''public int {name}(int a, int b) {{
        return a + b;
    }}''',
    '''public String {name}(String s) {{
        if (s == null) return null;
        return s.toUpperCase();
    }}''',
    '''public void {name}(List<Integer> list) {{
        list.removeIf(i -> i < 0);
    }}''',
    '''public boolean {name}(String email) {{
        return email != null && email.matches("^[A-Za-z0-9+_.-]+@[A-Za-z0-9.-]+$");
    }}''',
    '''public double {name}(double x) {{
        if (x == 0) throw new IllegalArgumentException("x cannot be zero");
        return 1 / x;
    }}''',
]

METHOD_NAMES = [
    "calculate", "process", "validate", "transform", "compute",
    "parse", "format", "convert", "extract", "merge",
    "filter", "sort", "search", "update", "delete"
]


def generate_mock_defects4j(
    num_samples: int = 100,
    output_dir: str = "data/defects4j/mock"
) -> List[Dict]:
    """
    生成模拟的 Defects4J 数据集
    
    Args:
        num_samples: 生成的样本对数（buggy + fixed = 2 * num_samples）
        output_dir: 输出目录
        
    Returns:
        List[Dict]: 训练数据列表
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    dataset = []
    
    for i in range(num_samples):
        template_idx = i % len(BUGGY_TEMPLATES)
        method_name = random.choice(METHOD_NAMES)
        
        # Buggy 版本
        buggy_code = BUGGY_TEMPLATES[template_idx].format(name=f"{method_name}_{i}")
        dataset.append({
            "id": f"bug_{i}",
            "code": buggy_code,
            "label": 1,  # 有 bug
            "project": "MockProject",
            "bug_id": str(i),
            "version": "buggy"
        })
        
        # Fixed 版本
        fixed_code = FIXED_TEMPLATES[template_idx].format(name=f"{method_name}_{i}")
        dataset.append({
            "id": f"fix_{i}",
            "code": fixed_code,
            "label": 0,  # 无 bug
            "project": "MockProject",
            "bug_id": str(i),
            "version": "fixed"
        })
    
    # 保存为 JSON
    output_file = output_path / "mock_dataset.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"生成 {len(dataset)} 条模拟数据")
    print(f"保存到: {output_file}")
    
    # 划分 train/val/test
    random.shuffle(dataset)
    n = len(dataset)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    splits = {
        'train': dataset[:train_end],
        'val': dataset[train_end:val_end],
        'test': dataset[val_end:]
    }
    
    for split_name, split_data in splits.items():
        split_file = output_path / f"{split_name}.json"
        with open(split_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        print(f"{split_name}: {len(split_data)} 条")
    
    return dataset


if __name__ == '__main__':
    # 生成 100 对样本（200 条数据）
    generate_mock_defects4j(num_samples=100)
