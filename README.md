# smart_knowledge_assistant

基于 LangChain + LangGraph 的智能知识库助手：本地文本 → FAISS 向量库 → **RAG 链式问答（入门版）** 或 **Agent + 多工具（进阶版 v0.2.0）**。

> 版本说明：`v0.1.0` 为纯 LCEL 链 + 会话记忆；`v0.2.0` 默认入口为 **Agent 模式**（可在 `main.py` 中切换 `use_agent`）。

## 🎯 项目概述

- 文档加载 → 切分 → OpenAI Embeddings → **FAISS 本地索引**
- **入门版**：`RunnableWithMessageHistory` 多轮对话 + RAG
- **进阶版**：`create_agent` + 工具（知识库问答 / 关键词检索 / 摘要）+ **Checkpointer**（LangGraph `MemorySaver`）

> 📝 配套博客（入门版）：[第6篇：实战项目-智能知识库助手（入门版）](https://blog.csdn.net/weixin_46253270/article/details/155783565)

> 📝 配套博客（进阶版）：[第11篇：实战项目-智能知识库助手（进阶版）](https://blog.csdn.net/weixin_46253270/article/details/156907380)


## ✨ 核心功能（v0.2.0）

### 进阶版（Agent）
- 🤖 **Agent 调度** - 根据意图选择工具
- 🔧 **query_knowledge_base** - RAG 理解型问答
- 🔎 **search_documents** - 检索片段 + 关键词高亮
- 📝 **summarize_document** - 主题摘要
- 🧠 **Checkpointer** - 线程级对话状态（`thread_id`）

### 入门版（LCEL）
- 💬 **多轮对话** - `session_id` 隔离历史
- 🗄️ **FAISS + RAG** - 与 v0.1.0 一致

## 🏗️ 技术架构

- **编排**：LangChain LCEL、`langchain.agents.create_agent`
- **状态**：LangGraph `MemorySaver`（进阶版）
- **LLM / Embedding**：OpenAI（`langchain-openai`）
- **向量库**：FAISS（`langchain-community`）
- **配置**：`core/config.py` 默认配置 + `merge_config`

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

编辑 `.env`，填入 `OPENAI_API_KEY`。可选：`OPENAI_BASE_URL`（兼容代理/兼容端点）。

### 3. 运行

默认 **进阶版（Agent）**：

```bash
python main.py
```

切换为 **入门版**：在 `main.py` 中将 `use_agent=True` 改为 `use_agent=False`，并按注释调整 `interactive_chat` 的 `session_id` / `thread_id`。

首次运行会生成 `faiss_index/`（已忽略）；再次运行会加载已有索引。

## 📚 使用说明

- 输入 `退出` / `quit` / `exit` 结束
- 输入 `重置` / `reset` 清空当前会话/线程记忆
- 替换知识库：修改 `main.py` 中 `load_documents([...])` 的路径或替换示例 `崔老道捉妖之夜闯董妃坟.txt`

## 📁 项目结构

```
smart_knowledge_assistant/
├── main.py
├── requirements.txt
├── .env.example
├── api/app.py
├── core/
│   ├── agent.py        # Agent（进阶版）
│   ├── tools.py        # 工具定义
│   ├── middleware.py   # 日志/性能/错误中间件
│   ├── memory.py       # 链式记忆 + Agent Checkpointer
│   ├── chain.py
│   ├── vectorstore.py
│   ├── loaders.py
│   └── config.py
└── test/
    ├── test_core.py
    ├── test_agent_tools.py
    └── test_clear_session.py
```

## 🛡️ 安全提示

- 禁止将真实密钥提交到仓库
- 仅加载**自己生成、可信**的 FAISS 索引（加载时使用 `allow_dangerous_deserialization=True`）

## 🧪 测试

```bash
python -m unittest -q
```

## 🤝 贡献指南

1. Fork → 新建分支 → 提交 PR

## 📄 许可证

MIT — 见仓库内 `LICENSE`（若有）。

## 📞 联系方式

- 仓库：[https://github.com/biao994/smart_knowledge_assistant](https://github.com/biao994/smart_knowledge_assistant)
- 作者：biao994
- 邮箱：zhengweibiao37@gmail.com

---

⭐ 若对你有帮助，欢迎 Star。
