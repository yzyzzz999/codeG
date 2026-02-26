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


def get_diff_files(buggy_dir: Path, fixed_dir: Path) -> List[str]:
    """用 git diff 找出修改的文件"""
    try:
        result = subprocess.run(
            ['git', 'diff', '--name-only', 'HEAD', str(fixed_dir)],
            cwd=str(buggy_dir),
            capture_output=True,
            text=True
        )
        files = result.stdout.strip().split('\n') if result.stdout else []
        return [f for f in files if f and f.endswith('.java')]
    except Exception as e:
        print(f"  git diff 失败: {e}")
        return []


def extract_changed_content(buggy_dir: Path, fixed_dir: Path) -> Optional[Tuple[str, str]]:
    """提取被修改的文件内容"""
    diff_files = get_diff_files(buggy_dir, fixed_dir)

    if not diff_files:
        # 如果没有 git diff 结果，尝试找 src 目录下的 Java 文件
        for src_pattern in ['src/main/java/**/*.java', 'src/java/**/*.java', 'src/**/*.java']:
            buggy_files = list(buggy_dir.glob(src_pattern))
            if buggy_files:
                # 找第一个在 fixed 中也存在的文件
                for buggy_file in buggy_files[:10]:  # 限制数量
                    rel_path = buggy_file.relative_to(buggy_dir)
                    fixed_file = fixed_dir / rel_path
                    if fixed_file.exists():
                        buggy_code = buggy_file.read_text(encoding='utf-8', errors='ignore')
                        fixed_code = fixed_file.read_text(encoding='utf-8', errors='ignore')
                        if buggy_code != fixed_code:
                            return buggy_code[:5000], fixed_code[:5000]  # 限制长度
                break

    # 用 git diff 的结果
    for file_path in diff_files[:3]:  # 最多取3个文件
        buggy_file = buggy_dir / file_path
        fixed_file = fixed_dir / file_path

        if buggy_file.exists() and fixed_file.exists():
            try:
                buggy_code = buggy_file.read_text(encoding='utf-8', errors='ignore')
                fixed_code = fixed_file.read_text(encoding='utf-8', errors='ignore')

                if buggy_code != fixed_code:
                    # 限制长度，避免 token 过长
                    return buggy_code[:5000], fixed_code[:5000]
            except:
                continue

    return None


def process_defects4j_pair(buggy_dir: Path, fixed_dir: Path) -> Optional[Dict]:
    """处理一对 buggy/fixed 项目"""
    print(f"  查找差异文件...")
    result = extract_changed_content(buggy_dir, fixed_dir)

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