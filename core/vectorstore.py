import os
import logging
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """向量存储管理器"""

    def __init__(self, config: dict):
        """
        初始化向量存储管理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.vectorstore: Optional[FAISS] = None
        self.embeddings = OpenAIEmbeddings(
            model=config.get("embedding_model", "text-embedding-3-small"),
            chunk_size=config.get("embedding_chunk_size", 200),
            timeout=config.get("embedding_timeout", 120)
        )


    def create_vector_store(self, documents: List[Document], save_path: str = None) -> None:
        """
        创建向量数据库

        Args:
            documents: 文档列表
            save_path: 向量数据库保存路径（可选）
        """
        save_path = save_path or self.config.get("vector_store_path", "./faiss_index")
        
        # 定义文档分割器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get("chunk_size", 500),
            chunk_overlap=self.config.get("chunk_overlap", 100),
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", ""]
        )

        # 文档分割
        texts = text_splitter.split_documents(documents)
        logger.info(f"文档分割完成，共{len(texts)}个文本块")   

        print(" 正在创建向量数据库，这可能需要几分钟时间...")
        
        # 将文档转换为向量嵌入并创建FAISS向量数据库
        self.vectorstore = FAISS.from_documents(texts, self.embeddings)

        # 保存索引
        self.vectorstore.save_local(save_path)
        logger.info(f"向量数据库已保存到 {save_path}")
        print(f" 向量数据库创建完成！")
    


    def load_vector_store(self, load_path: str = None) -> None:
        """
        加载已保存的向量数据库
        
        Args:
            load_path: 加载路径，如果为None则使用配置中的路径
            
        Raises:
            FileNotFoundError: 如果向量数据库不存在
        """
        load_path = load_path or self.config.get("vector_store_path", "./faiss_index")
        
        if not os.path.exists(f"{load_path}/index.faiss"):
            raise FileNotFoundError(f"向量数据库不存在: {load_path}")
        
        self.vectorstore = FAISS.load_local(
            load_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True  # 允许加载本地序列化数据，确保数据来源可信
        )
        logger.info(f"已加载向量数据库: {load_path}")

    def get_retriever(self):
        """
        获取检索器

        Returns:
            检索器对象

        Raises:
            ValueError: 如果向量数据库未初始化
        """
        if not self.vectorstore:
            raise ValueError("向量数据库未初始化，请先创建或初始化向量数据库")

        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.config.get("search_k", 5)}
        )

        