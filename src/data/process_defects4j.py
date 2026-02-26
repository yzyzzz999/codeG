"""
Defects4J 数据处理器
将检出的 buggy/fixed 项目转换为训练格式
"""

import json
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.parser.method_extractor import MethodExtractor


def extract_java_files(project_dir: Path) -> List[Path]:
    """提取目录中的所有 Java 文件"""
    java_files = []
    for src_dir in ["src/main/java", "src/java", "src"]:
        src_path = project_dir / src_dir
        if src_path.exists():
            java_files.extend(src_path.rglob("*.java"))
    return java_files


def extract_methods_from_project(project_dir: Path) -> List[Dict]:
    """从项目中提取所有方法"""
    extractor = MethodExtractor()
    java_files = extract_java_files(project_dir)

    all_methods = []
    for java_file in java_files:
        try:
            code = java_file.read_text(encoding='utf-8')
            methods = extractor.extract_methods_from_code(code)
            for method in methods:
                method_info = method.to_entity()
                method_info['file'] = str(java_file.relative_to(project_dir))
                all_methods.append(method_info)
        except Exception as e:
            print(f"  读取文件失败 {java_file}: {e}")

    return all_methods


def process_defects4j_pair(buggy_dir: Path, fixed_dir: Path) -> Dict:
    """
    处理一对 buggy/fixed 项目

    简化策略：取第一个文件的所有方法拼接
    """
    print(f"  提取 buggy 方法...")
    buggy_methods = extract_methods_from_project(buggy_dir)

    print(f"  提取 fixed 方法...")
    fixed_methods = extract_methods_from_project(fixed_dir)

    # 简化：取 buggy 和 fixed 的第一个方法作为样本
    # 实际应该用 diff 找出修改的方法，这里简化处理
    if buggy_methods and fixed_methods:
        return {
            "buggy_code": buggy_methods[0]['code'],
            "has_bug": 1,
            "fixed_code": fixed_methods[0]['code']
        }

    return None


def process_all_defects4j(data_dir: str, output_file: str):
    """处理所有 Defects4J 数据"""
    data_path = Path(data_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 找出所有 buggy 目录
    buggy_dirs = sorted(data_path.glob("*_buggy"))

    dataset = []
    for buggy_dir in buggy_dirs:
        bug_id = buggy_dir.name.replace("_buggy", "")
        fixed_dir = data_path / f"{bug_id}_fixed"

        if not fixed_dir.exists():
            print(f"跳过 {bug_id}: fixed 目录不存在")
            continue

        print(f"处理 {bug_id}...")
        sample = process_defects4j_pair(buggy_dir, fixed_dir)
        if sample:
            sample['bug_id'] = bug_id
            dataset.append(sample)
            print(f"  ✓ 成功提取")
        else:
            print(f"  ✗ 提取失败")

    # 保存数据集
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\n完成！共处理 {len(dataset)} 个样本")
    print(f"保存到: {output_path}")

    return dataset


if __name__ == '__main__':
    # 处理数据
    dataset = process_all_defects4j(
        data_dir="/codeG/data/defects4j/real",
        output_file="/codeG/data/defects4j/processed/defects4j_pairs.json"
    )