# CodeG - 企业级代码质量治理 Agent

基于大语言模型和 Agent 架构的智能代码审查系统，支持 Bug 检测、安全审计、性能优化建议。

## 核心能力

- **语义级代码理解**：基于 CodeBERT/CodeT5 的代码表征学习
- **智能 Bug 检测**：微调模型识别潜在缺陷（NPE、资源泄漏、并发问题等）
- **ReAct Agent 架构**：理解问题 → 检索上下文 → 分析推理 → 生成修复建议
- **GraphRAG 增强**：方法调用链分析，跨文件理解
- **生产级部署**：FastAPI + Docker + K8s，支持高并发

## 量化指标目标

| 指标 | 目标 | 验证方式 |
|------|------|----------|
| Bug 检出率 | >75% | Defects4J 数据集 |
| 误报率 | <20% | 人工标注验证 |
| 平均响应时间 | <3s | P99 |
| 并发支持 | 100 QPS | 压测 |
| 代码检索准确率 | >85% | Top-5 Recall |

## 技术栈

- **语言**: Python 3.10+
- **模型**: CodeT5, CodeBERT, 自研微调模型
- **Agent**: 原生 ReAct 实现 + LangChain 工具封装
- **向量数据库**: Milvus
- **图数据库**: Neo4j (GraphRAG)
- **API 框架**: FastAPI
- **部署**: Docker, Kubernetes
- **监控**: Prometheus + Grafana

## 项目结构

```
CodeG/
├── data/                   # 数据集和预处理
│   ├── raw/               # 原始数据
│   ├── processed/         # 处理后数据
│   └── defects4j/         # Defects4J 数据集
├── src/
│   ├── parser/            # Java 代码解析 (Tree-sitter)
│   ├── embedding/         # 代码向量化
│   ├── models/            # 模型定义和微调
│   ├── agent/             # ReAct Agent 核心
│   ├── rag/               # RAG 系统 (向量检索 + GraphRAG)
│   ├── api/               # FastAPI 服务
│   └── utils/             # 工具函数
├── configs/               # 配置文件
├── scripts/               # 训练和部署脚本
├── tests/                 # 单元测试和集成测试
├── docs/                  # 文档
├── docker/                # Docker 配置
└── notebooks/             # 实验和分析 notebook
```

## 开发阶段

- [x] Phase 1: 项目搭建与基础准备
- [ ] Phase 2: 代码解析与向量化
- [ ] Phase 3: Bug 检测模型微调
- [ ] Phase 4: Agent 框架实现
- [ ] Phase 5: RAG 与 GraphRAG
- [ ] Phase 6: 部署与优化

## 快速开始

```bash
# 克隆项目
git clone <repo-url>
cd CodeG

# 安装依赖
pip install -r requirements.txt

# 启动开发环境
docker-compose up -d

# 运行测试
pytest tests/
```

## License

MIT
