import logging
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from typing import List, Any, Optional, Dict

from .middleware import BaseMiddleware

logger = logging.getLogger(__name__)


class AgentManager:
    """Agent管理器"""
    def __init__(
        self,
        llm: ChatOpenAI,
        tools: List[BaseTool],
        checkpointer: Any,
        middlewares: Optional[List[BaseMiddleware]] = None
        ):
        

        """
        初始化Agent管理器

        Args:
            llm: 语言模型
            tools: 工具列表
            checkpointer: Checkpointer实例
            middlewares: 中间件列表（显式触发 on_agent_*）
        """
        self.llm = llm
        self.tools = tools
        self.checkpointer = checkpointer
        self.middlewares = middlewares or []
        self.agent = self._create_agent()

    def _create_agent(self):
        """
        创建Agent
        """
        agent = create_agent(
            model = self.llm,
            tools = self.tools,
            checkpointer = self.checkpointer,
            system_prompt = """你是一个智能知识库助手，专门帮助用户查询和分析知识库内容。

## 可用工具：
1. query_knowledge_base - 基于文档内容回答问题（适合"XX是什么"、"XX怎么样"等理解性问题）
2. search_documents - 精确搜索关键词（适合"搜索XX"、"查找XX"、"找XX这个词"等查找需求）
3. summarize_document - 生成指定主题的文档摘要（适合"总结XX"、"XX的概览"等需求）

## 工具选择规则（重要）：
- 如果用户明确说"搜索"、"查找"、"找"某个词 → 使用 search_documents
- 如果用户问"XX是什么"、"XX怎么样"、"XX的特点" → 使用 query_knowledge_base
- 如果用户要求"总结"、"概括"、"摘要" → 使用 summarize_document
"""
        )

        logger.info("Agent创建完成")
        return agent

    def invoke(self, input_text: str, thread_id: str="default"):
        """
        调用Agent

        Args:
            input_text:用户输入
            thread_id:线程ID（用户状态管理）

        Returns:
            Agent响应
        """

        for mw in self.middlewares:
            mw.on_agent_start({"input":input_text, "thread_id":thread_id} )

        config = {"configurable": {"thread_id":thread_id}}

        try:
            from langchain_core.messages import HumanMessage
            logger.debug(f"调用Agent,输入：{input_text[:50]}...")
            response = self.agent.invoke(
                {"messages": [HumanMessage(content=input_text)]},
                config=config
            )
            
            for mw in self.middlewares:
                response = mw.on_agent_end(response)
                logger.debug(f"Agent响应类型：{type(response)},响应键：{response.keys() if isinstance(response, dict) else 'N/A'}")
            return response
        except Exception as e:
            logger.error(f"Agent执行失败：{e}",exc_info=True)
            for mw in self.middlewares:
                mw.on_agent_error(e)
            raise

