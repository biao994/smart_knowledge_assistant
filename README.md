# smart_knowledge_assistant

一个最小可跑通的智能知识库助手：加载本地文本 → 构建 FAISS 向量库 → 基于检索增强生成（RAG）进行交互式问答。


## 🎯 项目概述

`smart_knowledge_assistant` 旨在提供一个“从零到可运行”的 RAG 小项目模板，适合学习/复现：

- 文档加载与清洗（文本、多编码兜底）
- 文档切分 → 向量化 → 本地向量库（FAISS）
- 检索增强生成（RAG Chain）
- 多会话的对话历史（v0.1.0 为链式模式）

> 📝 本项目有详细的开发教程博客，欢迎阅读：[第6篇：实战项目-智能知识库助手（入门版）](https://blog.csdn.net/weixin_46253270/article/details/155783565)

## ✨ 核心功能

### v0.1.0 功能特性
- 📄 **load_documents()** - 加载本地 `.txt` 文档（支持多编码兜底）
- 🧩 **split + embed** - 文档切分 + OpenAI Embeddings 向量化
- 🗄️ **FAISS 本地向量库** - 首次运行自动构建，后续自动加载
- 💬 **RAG 对话问答** - 基于检索到的上下文进行回答
- 🧠 **多轮对话记忆** - 基于 `RunnableWithMessageHistory` 的会话历史管理

## 🏗️ 技术架构

- **框架**：LangChain（链式编排）
- **LLM**：OpenAI Chat（通过环境变量配置）
- **Embedding**：OpenAI Embeddings
- **向量库**：FAISS（本地）
- **配置**：Python dict + 默认配置合并（`core/config.py`）

## 🚀 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
copy .env.example .env
```


### 3. 运行项目

```bash
python main.py
```

首次运行会在本地生成 `faiss_index/`

## 📚 使用示例

### 1. 启动交互式问答

```bash
python main.py
```

### 2. 常用命令

- 输入 `退出` / `quit` / `exit` 结束对话
- 输入 `重置` / `reset` 清空会话记忆

### 3. 替换知识库内容

在 `main.py` 中替换：

- `load_documents([...])` 传入你自己的 `.txt` 路径
- 或直接替换示例文本 `崔老道捉妖之夜闯董妃坟.txt`

## 📁 项目结构

```
smart_knowledge_assistant/
├── main.py                 # 入口：初始化、构建/加载向量库、启动交互式对话
├── requirements.txt        # 依赖
├── .env.example            # 环境变量示例（不要提交 .env）
├── api/
│   └── app.py              # SmartKnowledgeAssistant 主类
├── core/
│   ├── config.py           # 默认配置与合并
│   ├── loaders.py          # 文档加载（多编码兜底）
│   ├── vectorstore.py      # FAISS 向量库管理
│   ├── chain.py            # RAG Chain 构建
│   └── memory.py           # 会话历史管理
└── test/
    └── test_core.py        # 核心功能单测（可选）
```

## 🛡️ 安全特性 / 安全提示

- **密钥管理**：只提交 `.env.example`，禁止提交 `.env`
- **索引安全**：加载本地 FAISS 索引时允许反序列化，请仅加载自己生成、可信来源的索引文件

## 🔧 开发指南

### 运行单元测试（可选）

```bash
python -m unittest -q
```

## 🤝 贡献指南

欢迎提交 Issue 和 PR：

1. Fork 项目
2. 新建分支：`git checkout -b feature/xxx`
3. 提交更改：`git commit -m "feat: ..."`
4. 推送分支：`git push origin feature/xxx`
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 📞 联系方式

- 项目链接: [https://github.com/biao994/mcp-datatools](https://github.com/biao994/mcp-datatools)
- 作者: biao994
- 邮箱: zhengweibiao37@gmail.com

---

⭐ 如果这个项目对你有帮助，请给个 Star 支持一下！


