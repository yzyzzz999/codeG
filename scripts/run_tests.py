#!/usr/bin/env python3
"""
测试运行脚本
"""

import subprocess
import sys
import os
from pathlib import Path


def run_tests():
    """运行所有测试"""
    project_root = Path(__file__).parent.parent
    
    # 测试命令
    cmd = [
        "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--disable-warnings",
        "--cov=src",
        "--cov-report=html:coverage_html",
        "--cov-report=term-missing"
    ]
    
    print("🚀 运行 CodeG 测试套件...")
    print(f"项目目录: {project_root}")
    print(f"执行命令: {' '.join(cmd)}")
    print("-" * 50)
    
    # 切换到项目根目录
    original_cwd = Path.cwd()
    os.chdir(project_root)
    
    try:
        # 运行测试
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        print("-" * 50)
        if result.returncode == 0:
            print("✅ 所有测试通过!")
        else:
            print("❌ 测试失败!")
            
        return result.returncode
        
    finally:
        # 恢复原始工作目录
        os.chdir(original_cwd)


def run_unit_tests():
    """只运行单元测试"""
    cmd = ["pytest", "tests/unit/", "-v"]
    return subprocess.run(cmd).returncode


def run_specific_test(test_name):
    """运行特定测试"""
    cmd = ["pytest", "-v", "-k", test_name]
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "unit":
            exit(run_unit_tests())
        elif sys.argv[1] == "test":
            if len(sys.argv) > 2:
                exit(run_specific_test(sys.argv[2]))
            else:
                print("请指定测试名称")
                exit(1)
        else:
            print("用法: python run_tests.py [unit|test <test_name>]")
            exit(1)
    else:
        exit(run_tests())