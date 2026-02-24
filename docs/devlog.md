# CodeG 开发日志

## 2026-02-24 - Day 1

### 今日目标
- [x] 确定项目名称和定位
- [x] 搭建基础项目结构
- [ ] 创建 requirements.txt
- [ ] 初始化 Git 仓库
- [ ] 准备 Defects4J 数据集

### 关键决策
1. **项目名称**: CodeG (Code Guardian)
2. **定位**: 企业级代码质量治理平台，Java 为主，架构可扩展多语言
3. **向量数据库**: Milvus (企业级，支持高并发)
4. **Agent 实现**: 原生 ReAct + LangChain 工具封装

### 技术栈确认
- Python 3.10+
- PyTorch + Transformers
- Milvus (向量数据库)
- Neo4j (GraphRAG)
- FastAPI (API 服务)
- Tree-sitter (代码解析)

### 下一步
1. 完成项目骨架搭建
2. 配置开发环境
3. 下载 Defects4J 数据集
