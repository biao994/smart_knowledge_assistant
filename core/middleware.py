import logging
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)

class BaseMiddleware:

    def on_agent_start(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    def on_agent_end(self, outputs: Any) -> Any:
        return outputs

    def on_agent_error(self, error: Exception) -> None:
        return None

    def on_tool_start(self, tool_name: str, inputs:Dict[str, Any]) -> None:
        return None

    def on_tool_end(self, tool_name: str, outputs: Any) -> None:
        return None

    def on_tool_error(self, tool_name: str, error: Exception) -> None:
        return None

class LoggingMiddleware(BaseMiddleware):
    def on_agent_start(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"[agent] start: {inputs}")
        return inputs

    def on_agent_end(self, outputs: Any) -> Any:
        logger.info(f"[agent] end: {outputs}")
        return outputs

    def on_agent_error(self, error: Exception) -> None:
        logger.error(f"[agent] error: {error}")

    def on_tool_start(self, tool_name: str, inputs:Dict[str, Any]) -> None:
        logger.info(f"[tool] start: {tool_name}, {inputs}")

    def on_tool_end(self, tool_name: str, outputs: Any) -> None:
        logger.info(f"[tool] end: {tool_name}, {str(outputs)[:200]}")

    def on_tool_error(self, tool_name: str, error: Exception) -> None:
        logger.error(f"[tool] error: {tool_name}, {error}")

class ErrorHandlingMiddleware(BaseMiddleware):
    """错误处理中间件 - 统一记录异常"""

    def on_agent_error(self, error: Exception) -> None:
       logger.error(f"[agent] error: {error}")

    def on_tool_error(self, tool_name: str, error: Exception) -> None:
        logger.error(f"[tool] error: {tool_name}, {error}")

class PerformanceMiddleware(BaseMiddleware):
    """性能监控中间件 - 记录工具调用时间"""

    def __init__(self):
        self._start_times: Dict[str, float] = {}

    def on_tool_start(self, tool_name: str, inputs: Dict[str, Any]) -> None:
        # 记录工具开始执行的时间
        self._start_times[tool_name] = time.time()

    def on_tool_end(self, tool_name: str, output: Any) -> None:
        # 获取并移除开始时间
        start_time = self._start_times.pop(tool_name, None)
        if start_time is None:
            return

        # 计算消耗时间
        elapsed_time = time.time() - start_time
        logger.info(f"[tool] {tool_name} end, cost: {elapsed_time:.2f}s")
