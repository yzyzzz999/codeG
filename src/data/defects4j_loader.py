"""
Defects4J 数据集加载器
用于加载和处理 Defects4J 数据集进行训练和评估
"""

import json
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BugSample:
    """Bug 样本"""
    project: str
    bug_id: str
    buggy_code: str
    fixed_code: str
    bug_type: Optional[str] = None
    bug_location: Optional[str] = None


class Defects4JLoader:
    """Defects4J 数据加载器"""
    
    def __init__(self, data_dir: str = "data/defects4j"):
        self.data_dir = Path(data_dir)
        self.projects_dir = self.data_dir / "projects"
        
    def list_projects(self) -> List[str]:
        """列出可用的项目"""
        if not self.projects_dir.exists():
            return []
        return [d.name for d in self.projects_dir.iterdir() if d.is_dir()]
    
    def load_buggy_version(self, project: str) -> Optional[Path]:
        """加载 buggy 版本的项目路径"""
        project_path = self.projects_dir / project
        if project_path.exists():
            return project_path
        return None
    
    def extract_java_files(self, project_path: Path) -> List[Path]:
        """提取项目中的所有 Java 文件"""
        java_files = []
        src_dir = project_path / "src"
        if src_dir.exists():
            java_files.extend(src_dir.rglob("*.java"))
        return java_files
    
    def load_samples_from_project(self, project: str) -> List[BugSample]:
        """
        从项目中加载 bug 样本
        
        注意：这是简化版本，实际需要从 Defects4J 的 bug metadata 中解析
        """
        samples = []
        project_path = self.load_buggy_version(project)
        
        if not project_path:
            return samples
        
        java_files = self.extract_java_files(project_path)
        
        for java_file in java_files:
            try:
                code = java_file.read_text(encoding='utf-8')
                # 这里简化处理，实际应该解析 bug metadata
                samples.append(BugSample(
                    project=project,
                    bug_id=f"{project}_{java_file.stem}",
                    buggy_code=code,
                    fixed_code="",  # 需要从 fixed 版本获取
                    bug_type=None
                ))
            except Exception as e:
                print(f"读取文件失败 {java_file}: {e}")
                
        return samples


def prepare_defects4j_dataset(output_file: str = "data/processed/defects4j_dataset.json"):
    """
    准备 Defects4J 数据集
    
    将 Defects4J 项目转换为训练可用的格式
    """
    loader = Defects4JLoader()
    
    all_samples = []
    projects = loader.list_projects()
    
    print(f"发现 {len(projects)} 个项目: {projects}")
    
    for project in projects:
        print(f"处理项目: {project}")
        samples = loader.load_samples_from_project(project)
        all_samples.extend(samples)
        print(f"  - 提取 {len(samples)} 个样本")
    
    # 保存为 JSON
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([s.__dict__ for s in all_samples], f, indent=2, ensure_ascii=False)
    
    print(f"\n数据集已保存: {output_file}")
    print(f"总样本数: {len(all_samples)}")
    
    return all_samples


if __name__ == '__main__':
    # 测试
    loader = Defects4JLoader()
    projects = loader.list_projects()
    print(f"可用项目: {projects}")
    
    if projects:
        samples = loader.load_samples_from_project(projects[0])
        print(f"\n第一个项目的样本数: {len(samples)}")
        if samples:
            print(f"示例样本: {samples[0].bug_id}")
