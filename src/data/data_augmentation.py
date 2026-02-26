"""
CodeG - 数据增强
扩充训练数据量
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict


# 简单的变量名替换映射
VARIABLE_NAMES = ['x', 'y', 'z', 'a', 'b', 'c', 'i', 'j', 'k', 'n', 'm', 'temp', 'result', 'value']


def augment_by_variable_rename(code: str) -> str:
    """通过变量名替换增强数据"""
    # 简单替换常见的单字符变量名
    words = re.findall(r'\b[a-z]\b', code)
    if not words:
        return code

    # 随机选择一些变量替换
    unique_vars = list(set(words))
    if len(unique_vars) > 3:
        vars_to_replace = random.sample(unique_vars, 3)
    else:
        vars_to_replace = unique_vars

    new_code = code
    for var in vars_to_replace:
        new_var = random.choice([v for v in VARIABLE_NAMES if v != var])
        # 使用 word boundary 避免替换子串
        new_code = re.sub(rf'\b{var}\b', new_var, new_code)

    return new_code


def augment_by_code_truncation(code: str, max_lines: int = 50) -> str:
    """通过代码截断增强（取前N行）"""
    lines = code.split('\n')
    if len(lines) <= max_lines:
        return code
    return '\n'.join(lines[:max_lines])


def augment_by_whitespace(code: str) -> str:
    """通过调整空白字符增强"""
    # 随机添加或删除一些空行
    lines = code.split('\n')
    new_lines = []
    for line in lines:
        new_lines.append(line)
        if random.random() < 0.1:  # 10% 概率添加空行
            new_lines.append('')
    return '\n'.join(new_lines)


def augment_sample(sample: Dict, aug_types: List[str] = None) -> List[Dict]:
    """对一个样本进行多种增强"""
    if aug_types is None:
        aug_types = ['rename', 'truncate', 'whitespace']

    augmented = [sample]  # 保留原始样本

    code = sample['buggy_code']
    fixed = sample['fixed_code']

    # 变量名替换
    if 'rename' in aug_types:
        new_code = augment_by_variable_rename(code)
        new_fixed = augment_by_variable_rename(fixed)
        if new_code != code:
            augmented.append({
                'buggy_code': new_code,
                'has_bug': sample['has_bug'],
                'fixed_code': new_fixed,
                'bug_id': f"{sample['bug_id']}_rename"
            })

    # 代码截断
    if 'truncate' in aug_types:
        new_code = augment_by_code_truncation(code)
        new_fixed = augment_by_code_truncation(fixed)
        augmented.append({
            'buggy_code': new_code,
            'has_bug': sample['has_bug'],
            'fixed_code': new_fixed,
            'bug_id': f"{sample['bug_id']}_truncate"
        })

    # 空白调整
    if 'whitespace' in aug_types:
        new_code = augment_by_whitespace(code)
        new_fixed = augment_by_whitespace(fixed)
        augmented.append({
            'buggy_code': new_code,
            'has_bug': sample['has_bug'],
            'fixed_code': new_fixed,
            'bug_id': f"{sample['bug_id']}_whitespace"
        })

    return augmented


def augment_dataset(input_file: str, output_file: str, aug_types: List[str] = None):
    """增强整个数据集"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"原始数据: {len(data)} 条")

    augmented_data = []
    for sample in data:
        aug_samples = augment_sample(sample, aug_types)
        augmented_data.extend(aug_samples)

    # 去重（简单去重：基于代码内容）
    seen = set()
    unique_data = []
    for item in augmented_data:
        code_hash = hash(item['buggy_code'][:100])  # 取前100字符做hash
        if code_hash not in seen:
            seen.add(code_hash)
            unique_data.append(item)

    print(f"增强后: {len(unique_data)} 条")

    # 保存
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unique_data, f, indent=2, ensure_ascii=False)

    print(f"保存到: {output_file}")


if __name__ == '__main__':
    # 增强训练集
    augment_dataset(
        input_file="/codeG/data/defects4j/processed/train.json",
        output_file="/codeG/data/defects4j/processed/train_augmented.json",
        aug_types=['rename', 'truncate', 'whitespace']
    )