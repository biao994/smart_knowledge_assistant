import logging
from typing import Dict

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import Runnable


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



        