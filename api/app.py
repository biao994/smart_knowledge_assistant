"""主应用模块 - 智能知识库助手（支持入门版和进阶版）"""
import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from dotenv import load_dotenv

from core.config import merge_config
from core.loaders import load_documents
from core.vectorstore import VectorStoreManager
from core.memory import MemoryManager, AgentMemoryManager
from core.chain import build_rag_chain
from core.middleware import LoggingMiddleware, PerformanceMiddleware, ErrorHandlingMiddleware

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("SmartKnowledgeAssistant")
logger.setLevel(logging.INFO)


class SmartKnowledgeAssistant:
    """智能知识库助手（支持入门版和进阶版）"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, use_agent: bool = False):
        """
        初始化智能知识库助手
        
        Args:
            config: 用户配置字典，如果为None则使用默认配置
            use_agent: 是否使用Agent模式（True=进阶版，False=入门版）
        """
        # 加载配置
        self.config = merge_config(config)
        self.use_agent = use_agent
        
        # 初始化向量存储管理器
        self.vectorstore_manager: Optional[VectorStoreManager] = None
        
        # 初始化状态管理器
        self.memory_manager: Optional[MemoryManager] = None
        self.agent_memory_manager: Optional[AgentMemoryManager] = None
        
        # 初始化RAG链（入门版）
        self.chain = None
        
        # Agent相关组件（进阶版）
        self.tools_manager = None
        self.agent_manager = None
        
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
        if self.use_agent:
            # 进阶版：使用AgentMemoryManager（Checkpointer）
            self.agent_memory_manager = AgentMemoryManager(use_postgres=False)
            logger.info("使用Agent模式（进阶版）")
        else:
            # 入门版：使用MemoryManager（RunnableWithMessageHistory）
            self.memory_manager = MemoryManager()
            logger.info("使用LCEL链模式（入门版）")
        
        logger.info("智能知识库助手初始化完成")
    
    def load_documents(self, file_paths: list) -> list:
        """
        加载多个文档文件
        
        Args:
            file_paths: 文档路径列表

        Returns:
            成功加载的文档列表
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
        构建RAG链（带状态管理）- 入门版使用
        
        Args:
            session_id: 会话ID，用于区分不同会话
        """
        # 检查向量数据库是否已经初始化
        if not self.vectorstore_manager or not self.vectorstore_manager.vectorstore:
            raise ValueError("向量数据库未初始化，请先加载或创建向量数据库")

        # 获取检索器
        retriever = self.vectorstore_manager.get_retriever()

        # 构建基础的RAG链
        base_chain = build_rag_chain(
            retriever=retriever,
            config=self.config
        )

        # 使用RunnableWithMessageHistory包装链，实现多会话状态管理
        self.chain = self.memory_manager.wrap_chain_with_history(
            chain=base_chain,
            input_messages_key="input",
            history_messages_key="history"
        )
    
    def _build_agent_system(self):
        """构建Agent系统 - 进阶版使用"""
        if not self.vectorstore_manager or not self.vectorstore_manager.vectorstore:
            raise ValueError("向量数据库未初始化，请先加载或创建向量数据库")
        
        from langchain_openai import ChatOpenAI
        from core.tools import ToolsManager   
        from core.agent import AgentManager

        # 初始化LLM
        llm = ChatOpenAI(
            model=self.config.get("model", "gpt-3.5-turbo"),
            temperature=self.config.get("temperature", 0.1)
        )

        # 获取检索器
        retriever = self.vectorstore_manager.get_retriever()

        # 中间件
        middlewares = [
            LoggingMiddleware(),
            ErrorHandlingMiddleware(),
            PerformanceMiddleware(),
        ]

        # 创建工具管理器
        self.tools_manager = ToolsManager(
            retriever, 
            self.config, 
            llm,
            vectorstore=self.vectorstore_manager.vectorstore,
            middlewares=middlewares,
        )

        # 创建Agent管理器
        checkpointer = self.agent_memory_manager.get_checkpointer()
        self.agent_manager = AgentManager(
            llm=llm,
            tools=self.tools_manager.tools,
            checkpointer=checkpointer,
            middlewares=middlewares,
        )
        logger.info("Agent系统构建完成")
    
    def query(self, question: str, session_id: str = "default", thread_id: str = "default", max_retries: int = 3) -> str:
        """
        执行查询（支持两种模式）
        
        Args:
            question: 用户问题
            session_id: 会话ID（入门版使用）
            thread_id: 线程ID（进阶版使用）
            max_retries: 最大重试次数，默认3

        Returns:
            回答内容
        """
        #  检查问题是否为空
        if not question or not question.strip():
            return "问题不能为空"

        # 清理输入
        question = question.strip()

        if not self.vectorstore_manager or not self.vectorstore_manager.vectorstore:
            return "向量数据库未初始化，请先加载或创建向量数据库"

        if self.use_agent:
            return self._query_with_agent(question, thread_id, max_retries)
        else:
            return self._query_with_chain(question, session_id, max_retries)
        
    def _query_with_chain(self, question: str, session_id: str, max_retries: int = 3) -> str:
        """使用LCEL链查询（入门版）"""
        # 构建链（延迟构建）
        if not self.chain:
            self._build_chain(session_id)

        # 重试机制，防止网络波动等问题导致查询失败
        for attempt in range(max_retries):
            try:
                run_config = {"configurable": {"session_id": session_id}}
                response = self.chain.invoke(
                    {"input": question},
                    config=run_config
                )
                return response
            except Exception as e:
                logger.error(f"查询失败（尝试{attempt + 1}/{max_retries}）: {e}")
                if attempt == max_retries - 1:
                    return f"抱歉，查询过程中出现错误：{str(e)}"  
        return response

    def _query_with_agent(self, question: str, thread_id: str = "default", max_retries: int = 3) -> str:
        """
        使用Agent查询
        
        Args:
            question: 用户问题
            thread_id: 线程ID（进阶版使用）
            
        Returns:
            回答内容
        """
        if not self.agent_manager:
            self._build_agent_system()

        # 重试机制
        for attempt in range(max_retries):
            try:
                response = self.agent_manager.invoke(question, thread_id)

                if isinstance(response, dict) and "messages" in response and response["messages"]:
                    # 提取最后一条AIMessage的内容
                    last_message = response["messages"][-1]
                    if last_message.content:
                        return last_message.content
                    else:
                        logger.warning("Agent返回空回答")
                        return "Agent返回空回答"
                else:
                    logger.warning("Agent返回无效格式")
                    return "Agent返回无效格式"
            except Exception as e:
                logger.error(f"查询失败（尝试{attempt + 1}/{max_retries}）: {e}")
                if attempt == max_retries - 1:
                    return f"抱歉，查询过程中出现错误：{str(e)}"  
        return response
    
    def clear_session(self, session_id: str = "default", thread_id: str = "default") -> None:
        """
        清空指定会话/线程的历史记录
        
        Args:
            session_id: 会话ID（入门版使用）
            thread_id: 线程ID（进阶版使用）
        """
        if self.use_agent:
            # 进阶版：重建 checkpointer + 丢弃 agent，避免残留未配对的 tool_calls 影响后续请求
            self.agent_memory_manager = AgentMemoryManager(use_postgres=False)
            self.agent_manager = None
            self.tools_manager = None
            logger.info(f"线程 {thread_id} 的历史记录已清除")
        else:
            # 入门版：清空历史并丢弃已包装的链，确保下次按 session_id 重新构建
            if self.memory_manager:
                self.memory_manager.clear_session(session_id)
            self.chain = None
            logger.info(f"会话 {session_id} 的历史记录已清除")
    
    def interactive_chat(self, session_id: str = "default", thread_id: str = "default"):
        """
        交互式对话界面
        
        Args:
            session_id: 会话ID（入门版使用）
            thread_id: 线程ID（进阶版使用）
        """
        mode_name = "进阶版（Agent模式）" if self.use_agent else "入门版（LCEL链模式）"
        print(f"🚀 智能知识库助手（{mode_name}）已启动！")
        print("📚 功能特性：")
        if self.use_agent:
            print("  • Agent智能决策")
            print("  • 多工具动态调度")
            print("  • Checkpointer状态管理")
            print(f"  • 当前线程ID: {thread_id}")
        else:
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
                    self.clear_session(session_id, thread_id)
                    print("  对话记忆已重置")
                    continue
                elif not user_input:
                    continue

                print(" 助手：", end="", flush=True)

                # 添加性能监控
                start_time = datetime.now()
                if self.use_agent:
                    response = self.query(user_input, thread_id=thread_id)
                else:
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
                import traceback
                traceback.print_exc()
