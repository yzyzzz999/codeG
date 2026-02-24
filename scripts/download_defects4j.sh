#!/bin/bash
# Defects4J 数据集下载脚本

set -e

DATA_DIR="$(dirname "$0")/../data/defects4j"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=== 下载 Defects4J 数据集 ==="

# 克隆 Defects4J 仓库
if [ ! -d "defects4j" ]; then
    echo "克隆 Defects4J 仓库..."
    git clone https://github.com/rjust/defects4j.git
fi

cd defects4j

# 初始化 Defects4J
echo "初始化 Defects4J..."
./init.sh

# 创建项目目录
mkdir -p ../projects

echo "=== 下载常用项目 ==="

# 下载几个常用项目（轻量级）
PROJECTS=("Lang" "Math" "Chart")

for project in "${PROJECTS[@]}"; do
    echo "检出 $project..."
    defects4j checkout -p "$project" -v "1b" -w "../projects/${project}_1"
done

echo "=== 完成 ==="
echo "数据集位置: $DATA_DIR/projects"
echo ""
echo "项目列表:"
ls -la ../projects/
