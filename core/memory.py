import logging
from typing import Dict, Optional

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import Runnable
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

class MemoryManager:
    """状态管理器 - 使用RunnableWithMessageHistory 实现多会话支持"""

    def __init__(self):
        """初始化状态管理器"""
        # 存储不同会话的历史记录
        self.store: Dict[str, ChatMessageHistory] = {}

    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        """
        获取或创建会话历史记录

        Args:
            session_id: 会话ID

        Returns:
            ChatMessageHistory对象
        """
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
            logger.info(f"创建新会话 {session_id}")

        return self.store[session_id]
        

    def clear_session(self, session_id: str) -> None:
        """
        清除指定会话的历史记录
        
        Args:
            session_id: 会话ID
        """
        if session_id in self.store:
            self.store[session_id].clear()
            logger.info(f"清除会话 {session_id} 的历史记录")

    def wrap_chain_with_history(
        self,
        chain: Runnable,
        input_messages_key: str = "input",
        history_messages_key: str = "history"
    ) -> RunnableWithMessageHistory:
        """使用RunnableWithMessageHistory 包装链，使其具备自动管理历史的能力
        
        Args:
            chain: 要包装的Runnable链
            input_messages_key: 输入消息键名，默认"input"
            history_messages_key: 历史消息键名，默认"history"

        Returns:
            包装后的链
        """
        return RunnableWithMessageHistory(
            runnable=chain,
            get_session_history=self.get_session_history,
            input_messages_key=input_messages_key,
            history_messages_key=history_messages_key,
        )

    def get_all_sessions(self) -> list:
        """
        获取所有会话ID列表
        """
        return list(self.store.keys())

class AgentMemoryManager:
    """Agent状态管理器 - 使用Checkpointer实现Agent的状态管理（进阶版）"""
    def __init__(self, use_postgres: bool = False, postgres_url: Optional[str] = None):
        """
        初始化Agent状态管理器

        Args:
            use_postgres: 是否使用PostgreSQL持久化（生产环境）
            postgres_url: PostgreSQL连接URL
        """

        if use_postgres and postgres_url:
            # 生产环境：使用PostgreSQL持久化
            try:
                from langchain_postgres import PostgresSaver
                self.checkpointer = PostgresSaver.from_conn_string(postgres_url)
                logger.info("使用PostgreSQL进行状态持久化")
            except ImportError:
                logger.warning("未安装langchain-postgres，使用内存存储")
            except Exception as e:
                logger.error(f"PostgreSQL初始化失败: {e}")
                self.checkpointer = MemorySaver()
        else:
            # 开发环境：使用内存存储
            self.checkpointer = MemorySaver()
            logger.info("使用内存存储进行状态持久化")

    def get_checkpointer(self):
        """
        获取Checkpointer实例
        """
        return self.checkpointer
