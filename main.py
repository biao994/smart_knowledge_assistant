"""主入口文件 - 智能知识库助手（支持入门版和进阶版）"""
import os
import sys
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.app import SmartKnowledgeAssistant
from core.config import merge_config


def main():
    """主函数 - 初始化并启动智能知识库助手"""
    try:
        # 创建助手实例（进阶版 - Agent模式）
        # 如果要使用入门版，设置 use_agent=False
        config = {
            "chunk_size": 200,
            "chunk_overlap": 50,
            "search_k": 3,
            "memory_window": 10,
            "model": "gpt-3.5-turbo"
        }
        
        # 使用Agent模式（进阶版）
        assistant = SmartKnowledgeAssistant(config, use_agent=True)
        
        # 检查是否已有向量数据库
        vector_store_path = "./faiss_index"
        if os.path.exists(f"{vector_store_path}/index.faiss"):
            print("🔍 发现已存在的向量数据库，正在加载...")
            assistant.load_vector_store(vector_store_path)
        else:
            print("📖 未找到现有向量数据库，正在创建新的...")
            # 加载文档（支持多个文件）
            documents = assistant.load_documents([
                "崔老道捉妖之夜闯董妃坟.txt",
            ])
            
            # 创建向量数据库
            assistant.create_vector_store(documents, vector_store_path)

        # 启动对话（Agent模式使用thread_id，入门版使用session_id）
        from datetime import datetime

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        thread_id = f"thread_{run_id}"
        session_id = f"session_{run_id}"

        assistant.interactive_chat(
            thread_id=thread_id if assistant.use_agent else "default",
            session_id=session_id if not assistant.use_agent else "default",
        )  

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
