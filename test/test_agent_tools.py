"""test_agent_tools.py"""
import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from core.middleware import LoggingMiddleware, ErrorHandlingMiddleware, PerformanceMiddleware
from core.tools import ToolsManager
class _FakeRetriever:
    def invoke(self, query:str):
        return [
            Document(page_content=f"内容片段：{query}", metadata={"source": "fake.txt"}),
            Document(page_content="另一段内容", metadata={"source": "fake.txt"})
        ]

class _FakeLLMResponse:
    def __init__(self, content:str):
        self.content = content

class _FakeLLM:
    def invoke(self, prompt):
        return _FakeLLMResponse(content=f"摘要结果：{prompt[:20]}")

class _FakeChain:
    def invoke(self, inputs):
        # tools.py 里调用 rag_chain.invoke({"input": question, "history": []})
        question = inputs.get("input", "")
        return f"RAG回答：{question}"

class TestAgentTools(unittest.TestCase):
    @patch("core.chain.build_rag_chain", return_value=_FakeChain())
    def test_tools_manager_creates_three_named_tools(self, mock_build_rag_chain):
        tm = ToolsManager(
            retriever=_FakeRetriever(),
            config={"model": "gpt-3.5-turbo"},
            llm=_FakeLLM(),
            vectorstore=None,
            middlewares=[LoggingMiddleware(), ErrorHandlingMiddleware(), PerformanceMiddleware()]
        )
        
        self.assertEqual(len(tm.tools), 3)

        names = [getattr(t, "name", None) for t in tm.tools]
        self.assertEqual(set(names), {"query_knowledge_base", "search_documents", "summarize_document"})

    @patch("core.chain.build_rag_chain", return_value=_FakeChain())
    def test_tools_can_run_offline(self, mock_build_rag_chain):
        tm = ToolsManager(
            retriever=_FakeRetriever(),
            config={"model": "gpt-3.5-turbo"},
            llm=_FakeLLM(),
            vectorstore=None,
            middlewares=[LoggingMiddleware(), ErrorHandlingMiddleware(), PerformanceMiddleware()]
        )

        tools_by_name = {t.name: t for t in tm.tools}

        out1 = tools_by_name["query_knowledge_base"].invoke({"question":"崔老道是谁"})
        self.assertIsInstance(out1, str)
        self.assertIn("崔老道", out1)

        out2 = tools_by_name["search_documents"].invoke({"keyword": "崔老道"})
        self.assertIsInstance(out2, str)
        self.assertIn("崔老道", out2)

        out3 = tools_by_name["summarize_document"].invoke({"topic": "RAG"})
        self.assertIsInstance(out3, str)
        self.assertIn("RAG", out3)


if __name__ == "__main__":
    unittest.main()
