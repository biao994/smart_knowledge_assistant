"""文档加载模块"""
import os
import logging
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_documents(file_paths: List[str]) -> List[Document]:
    """
    加载多个文档文件
    
    Args:
        file_paths: 文件路径列表
        
    Returns:
        文档列表
        
    Raises:
        ValueError: 如果没有成功加载任何文档
    """
    all_documents = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            logger.warning(f"文件不存在: {file_path}")
            continue
        
        try:

            # 尝试多种编码方式
            encodings = ['utf-8','gbk', 'gb2312', 'ansi']
            documents = None
            
            for encoding in encodings:
                try:
                    loader = TextLoader(file_path, encoding=encoding)
                    documents = loader.load()
                    logger.info(f"成功使用 {encoding} 编码加载文件: {file_path}")
                    break
                except UnicodeDecodeError:
                    continue
            
            if documents is None:
                raise ValueError(f"无法使用任何编码加载文件: {file_path}")

            
            # 添加文件来源信息
            for doc in documents:
                doc.metadata["source"] = file_path
            
            all_documents.extend(documents)
            logger.info(f"成功加载文件: {file_path}, 文档数: {len(documents)}")
            
        except Exception as e:
            logger.error(f"加载文件失败 {file_path}: {e}")
            continue
    
    if not all_documents:
        raise ValueError("没有成功加载任何文档")
    
    return all_documents