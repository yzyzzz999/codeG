"""
Defects4J 数据处理器
将检出的 buggy/fixed 项目转换为训练格式
"""

import json
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def get_diff_files(buggy_dir: Path, bug_id: str) -> List[str]:
    """用 git diff 找出修改的文件"""
    try:
        # 从 bug_id 提取项目名和编号，如 Lang_5 -> D4J_Lang_5
        tag_prefix = f"D4J_{bug_id}"

        result = subprocess.run(
            ['git', 'diff', f'{tag_prefix}_BUGGY_VERSION', f'{tag_prefix}_FIXED_VERSION', '--name-only'],
            cwd=str(buggy_dir),
            capture_output=True,
            text=True
        )
        files = result.stdout.strip().split('\n') if result.stdout else []
        return [f for f in files if f and f.endswith('.java')]
    except Exception as e:
        print(f"  git diff 失败: {e}")
        return []


def extract_changed_content(buggy_dir: Path, fixed_dir: Path, diff_files: List[str]) -> Optional[Tuple[str, str]]:
    """提取被修改的文件内容"""
    for file_path in diff_files:
        buggy_file = buggy_dir / file_path
        fixed_file = fixed_dir / file_path

        if buggy_file.exists() and fixed_file.exists():
            try:
                buggy_code = buggy_file.read_text(encoding='utf-8', errors='ignore')
                fixed_code = fixed_file.read_text(encoding='utf-8', errors='ignore')

                if buggy_code != fixed_code:
                    # 限制长度，避免 token 过长
                    return buggy_code[:5000], fixed_code[:5000]
            except Exception as e:
                print(f"  读取文件失败 {file_path}: {e}")
                continue

    return None


def process_defects4j_pair(buggy_dir: Path, fixed_dir: Path, bug_id: str) -> Optional[Dict]:
    """处理一对 buggy/fixed 项目"""
    print(f"  查找差异文件...")
    diff_files = get_diff_files(buggy_dir, bug_id)

    if not diff_files:
        print(f"  没有找到修改的 Java 文件")
        return None

    print(f"  修改的文件: {diff_files}")

    result = extract_changed_content(buggy_dir, fixed_dir, diff_files)

    if result:
        buggy_code, fixed_code = result
        return {
            "buggy_code": buggy_code,
            "has_bug": 1,
            "fixed_code": fixed_code
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
        sample = process_defects4j_pair(buggy_dir, fixed_dir, bug_id)
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