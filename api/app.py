"""主应用模块 - 智能知识库助手（初版设计）"""
import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from dotenv import load_dotenv

from core.config import merge_config, get_default_config
from core.loaders import load_documents
from core.vectorstore import VectorStoreManager
from core.memory import MemoryManager
from core.chain import build_rag_chain


# 配置日志
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("SmartKnowledgeAssistant")
logger.setLevel(logging.INFO)


class SmartKnowledgeAssistant:
    """智能知识库助手（初版设计）"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化智能知识库助手
        
        Args:
            config: 用户配置字典，如果为None则使用默认配置
        """
        # 加载配置
        self.config = merge_config(config)
        
        # 初始化向量存储管理器
        self.vectorstore_manager: Optional[VectorStoreManager] = None
        
        # 初始化状态管理器
        self.memory_manager: Optional[MemoryManager] = None
        
        # 初始化RAG链
        self.chain = None
        
        # 初始化环境
        self._setup()
    
    def _setup(self):
        """初始化所有组件"""
        # 加载环境变量
        load_dotenv()
        
        # 检查API密钥
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("未找到 OPENAI_API_KEY，请检查.env文件")
        
        # 初始化向量存储管理器
        self.vectorstore_manager = VectorStoreManager(self.config)
        
        # 初始化状态管理器
        self.memory_manager = MemoryManager()
        
        logger.info("智能知识库助手初始化完成")
    
    def load_documents(self, file_paths: list) -> list:
        """
        加载多个文档文件
        
        Args:
            file_paths: 文档路径列表

        Returns:
            成功加载的文档路径列表
        """
        # 直接调用核心模块的文档加载函数
        return load_documents(file_paths)
        
    
    def create_vector_store(self, documents: list, save_path: str = None) -> None:
        """
        创建向量数据库
        
        Args:
            documents: 文档列表
            save_path: 保存路径，如果为None则使用配置中的路径
        """
        # 调用向量存储管理器的创建方法
        self.vectorstore_manager.create_vector_store(documents, save_path)
    
    def load_vector_store(self, save_path: str = None) -> None:
        """
        加载已保存的向量数据库
        
        Args:
            save_path: 保存路径，如果为None则使用配置中的路径
        """
        # 调用向量存储管理器的加载方法
        self.vectorstore_manager.load_vector_store(save_path)
    
    def _build_chain(self, session_id: str = "default") -> None:
        """
        构建RAG链 （带状态管理）

        Args:
            session_id: 会话ID，用于区分不同会话

        """
        # 检查向量数据库是否已经初始化
        if not self.vectorstore_manager or not self.vectorstore_manager.vectorstore:
            raise ValueError("向量数据库未初始化，请先加载或创建向量数据库")

        # 获取检索器
        retriever = self.vectorstore_manager.get_retriever(
        )

        # 构建基础的RAG链
        base_chain = build_rag_chain(
            retriever=retriever,
            config=self.config
        )

        # 使用RunnableWithMessageHistory包装链， 实现多会话状态管理
        self.chain = self.memory_manager.wrap_chain_with_history(
            chain=base_chain,
            input_messages_key="input",
            history_messages_key="history"
        )

    def query(self, question: str, session_id: str = "default", max_retries: int = 3) -> str:
        """
        执行查询 （带重试机制）
        
        Args:
            question: 用户问题
            session_id: 会话ID，用于区分不同会话
            max_retries: 最大重试次数，默认3


        Returns:
            回答内容
        """
        #  检查问题是否为空
        if not question or not question.strip():
            return "问题不能为空"

        # 清理输入
        question = question.strip()

        # 构建链（延迟构建，只有在第一次查询时才构建）
        if not self.chain:
            self._build_chain(session_id)
        
        # 重试机制，防止网络波动等问题导致查询失败
        for attempt in range(max_retries):
            try:
                run_config = {"configurable":{"session_id":session_id}}
                response = self.chain.invoke(
                    {"input": question},
                    config=run_config
                )
                return response
            except Exception as e:
                logger.error(f"查询失败 （尝试{attempt + 1} / {max_retries}）: {e}")
                if attempt == max_retries - 1:
                    return f"抱歉，查询过程中出现错误：{str(e)}"       
    
    def clear_session(self, session_id: str = "default") -> None:
        """
        清空指定会话的历史记录
        
        Args:
            session_id: 会话ID
        """
        # 调用状态管理器的清除方法
        self.memory_manager.clear_session(session_id)
        logger.info(f"会话 {session_id} 的历史记录已清除")
    
    def interactive_chat(self, session_id: str = "default"):
        """
        交互式对话界面
        
        Args:
            session_id: 会话ID，用于区分不同会话
        """
        print("🚀 智能知识库助手已启动！")
        print("📚 功能特性：")
        print("  • 多轮对话记忆")
        print("  • 智能检索") 
        print(f"  • 当前会话ID: {session_id}")
        print("输入'退出'结束对话，'重置'清空记忆")
        print("-" * 60)

        while True:
            try:
                user_input = input("用户: ").strip()
                if user_input.lower() in ['退出', 'quit', 'exit']:
                    print(" 感谢使用！")
                    break
                elif user_input.lower() in ['重置', 'reset']:
                    # 重置对话记忆
                    self.clear_session(session_id)
                    print("  对话记忆已重置")
                    continue
                elif not user_input:
                    continue

                print(" 助手：", end="", flush=True)

                # 添加性能监控
                start_time = datetime.now()
                response = self.query(user_input, session_id)
                duration = (datetime.now() - start_time).total_seconds()

                print(response)
                print(f"  响应时间: {duration:.2f}秒")

            except KeyboardInterrupt:
                print("\n  对话已结束")
                break
            except Exception as e:
                print(f"\n 系统错误: {e}")
                logger.error(f"交互式聊天出错: {e}")



