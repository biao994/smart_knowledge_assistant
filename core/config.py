from typing import Dict, Any 

def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    return {
        "chunk_size": 500,  # 文档分块大小
        "chunk_overlap": 100,  # 分块重叠大小
        "search_k": 5,  # 检索返回的相关文档数量
        "memory_window": 10,  # 记忆窗口大小
        "temperature": 0.1,  # 语言模型温度参数，控制生成文本的随机性，值越小输出越确定
        "model": "gpt-3.5-turbo",  # 使用的语言模型
        "embedding_model": "text-embedding-3-small",  # 文本嵌入模型，用于将文本转换为向量表示
        "embedding_chunk_size": 200,  # 嵌入处理分块大小
        "embedding_timeout": 120,  # 嵌入请求超时时间（秒）
        "llm_timeout": 30,  # 语言模型请求超时时间（秒）
        "llm_max_retries": 3,  # 语言模型最大重试次数
        "vector_store_path": "./faiss_index",  # 向量数据库存储路径
        "max_tokens": 1000  # 生成的最大token数量
    }


def merge_config(user_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """合并用户配置和默认配置"""
    default_config = get_default_config()
    if user_config is None:
        return default_config
    return {**default_config, **user_config}