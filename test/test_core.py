import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import get_default_config, merge_config
from core.loaders import load_documents
from core.vectorstore import VectorStoreManager
from core.memory import MemoryManager
from core.chain import build_rag_chain


"""核心功能测试模块"""
class TestCoreFunctions(unittest.TestCase):
    """核心功能测试类"""

    def setUp(self):
        """测试前准备"""
        self.config = get_default_config()
        self.test_file = "test_document.txt"

        # 创建测试文档
        with open(self.test_file, 'w', encoding='utf-8') as f:
            f.write("LangChain是一个用于开发由语言模型驱动的应用程序的框架。\n")
            f.write("它允许开发者将LLM与外部数据源和计算连接起来。\n")

    def tearDown(self):
        """测试后清理"""
        # 在每个测试方法执行后都会运行这个方法
        # 清理测试过程中创建的文件
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_config_functions(self):
        """测试配置函数"""
        # 首先测试默认配置是否正常加载
        default_config = get_default_config()
        self.assertIn("chunk_size", default_config)
        self.assertIn("model", default_config)

        # 然后测试配置合并功能
        user_config = {"model":"gpt-4", "temperature":0.5}
        merged = merge_config(user_config)

        # 检查用户配置是否正确覆盖默认配置
        self.assertEqual(merged["model"], "gpt-4")
        self.assertEqual(merged["temperature"], 0.5)
        # 检查未指定的配置项是否保持默认值
        self.assertEqual(merged["chunk_size"], default_config["chunk_size"])


    def test_document_loading(self):
        """测试文档加载"""
        # 测试是否正确加载我们创建的测试文档
        documents = load_documents([self.test_file])
        # 应该成功加载一个文档
        self.assertEqual(len(documents), 1)
        # 文档内容应该包含我们写入的文本
        self.assertIn("LangChain", documents[0].page_content)



    @patch('core.vectorstore.OpenAIEmbeddings')
    def test_vectorstore_manager(self, mock_embeddings):
        """测试向量存储管理器"""
        
        # 由于我们不想在测试中真的调用OpenAI API，所以使用mock来模拟嵌入模型
        mock_embeddings.return_value = MagicMock()
        
        # 创建向量存储管理器实例
        manager = VectorStoreManager(self.config)
        
        # 检查嵌入模型是否正确初始化
        self.assertIsNotNone(manager.embeddings)
        
        # 检查配置是否正确传递 
        self.assertEqual(manager.config, self.config)
        
        # 验证mock_embeddings被调用
        mock_embeddings.assert_called_once()
        
        
    def test_memory_manager(self):
        """测试状态管理器"""
        # 创建状态管理器实例

        memory_manager = MemoryManager()

        # 测试会话创建功能
        # 同一个会话ID应该返回相同的ChatMessageHistory对象
        history1 = memory_manager.get_session_history("session1")
        history2 = memory_manager.get_session_history("session1")
        self.assertEqual(history1, history2)
        
        # 不同的会话ID应该返回不同的ChatMessageHistory对象
        history3 = memory_manager.get_session_history("session2")
        self.assertIsNot(history1, history3)

        # 测试获取所有会话ID的功能
        sessions = memory_manager.get_all_sessions()
        self.assertIn("session1", sessions)
        self.assertIn("session2", sessions)

        # 测试清空会话功能
        memory_manager.clear_session("session1")

        # 会话ID应该仍然存在，当历史记录已被清空
        self.assertIn("session1", memory_manager.get_all_sessions())

if  __name__ == "__main__":
    unittest.main()
