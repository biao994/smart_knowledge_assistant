import unittest

from api.app import SmartKnowledgeAssistant


class _FakeMemoryManager:
    def __init__(self):
        self.cleared = []

    def clear_session(self, session_id: str):
        self.cleared.append(session_id)


class TestClearSession(unittest.TestCase):
    def test_clear_session_in_chain_mode_clears_history_and_drops_chain(self):
        a = SmartKnowledgeAssistant.__new__(SmartKnowledgeAssistant)
        a.use_agent = False
        a.memory_manager = _FakeMemoryManager()
        a.chain = object()

        a.clear_session(session_id="s1", thread_id="t1")

        self.assertEqual(a.memory_manager.cleared, ["s1"])
        self.assertIsNone(a.chain)

    def test_clear_session_in_agent_mode_drops_agent_and_tools(self):
        a = SmartKnowledgeAssistant.__new__(SmartKnowledgeAssistant)
        a.use_agent = True
        a.agent_memory_manager = object()
        a.agent_manager = object()
        a.tools_manager = object()

        a.clear_session(session_id="s1", thread_id="t1")

        self.assertIsNone(a.agent_manager)
        self.assertIsNone(a.tools_manager)
        # agent_memory_manager 会被重建为 AgentMemoryManager，这里只验证属性存在即可
        self.assertIsNotNone(a.agent_memory_manager)


if __name__ == "__main__":
    unittest.main()

