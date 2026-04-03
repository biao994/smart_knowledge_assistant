"""工具模块 - 封装知识库相关工具供Agent调用"""
import logging
import re
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI

from .middleware import BaseMiddleware

logger = logging.getLogger(__name__)


class ToolsManager:
    """工具管理器 - 专注于知识库相关工具"""
    
    def __init__(
        self,
        retriever: BaseRetriever,
        config: dict,
        llm: ChatOpenAI,
        vectorstore=None,
        middlewares: Optional[List[BaseMiddleware]] = None,
    ):
        """
        初始化工具管理器
        
        Args:
            retriever: 文档检索器
            config: 配置字典
            llm: 语言模型
            vectorstore: 向量存储对象（可选，用于搜索工具）
            middlewares: 中间件列表（教学版：显式触发 on_tool_*）
        """
        self.retriever = retriever
        self.config = config
        self.llm = llm
        self.vectorstore = vectorstore
        self.middlewares = middlewares or []
        self.tools = self._create_tools()

    def _mw_tool_start(self, tool_name: str, inputs: Dict[str, Any]) -> None:
        for mw in self.middlewares:
            mw.on_tool_start(tool_name, inputs)

    def _mw_tool_end(self, tool_name: str, output: Any) -> None:
        for mw in self.middlewares:
            mw.on_tool_end(tool_name, output)

    def _mw_tool_error(self, tool_name: str, error: Exception) -> None:
        for mw in self.middlewares:
            mw.on_tool_error(tool_name, error)
    
    def _create_tools(self):
        """创建知识库相关工具列表"""
        tools = []
        
        # 1. RAG知识库查询工具
        rag_tool = self._create_rag_tool()
        tools.append(rag_tool)
        
        # 2. 文档关键词搜索工具
        search_tool = self._create_search_tool()
        tools.append(search_tool)
        
        # 3. 文档摘要工具
        summary_tool = self._create_summary_tool()
        tools.append(summary_tool)
        
        logger.info(f"已创建 {len(tools)} 个知识库工具")
        return tools
    
    def _create_rag_tool(self):
        """创建RAG知识库查询工具"""
        from .chain import build_rag_chain
        
        # 构建RAG链（注意：这里需要构建一个简化版的RAG链，不包含历史记录）
        rag_chain = build_rag_chain(self.retriever, self.config)
        
        @tool
        def query_knowledge_base(question: str) -> str:
            """基于文档内容回答问题，适合需要理解和解释的问题。
            
            使用场景：
            - 用户问"XX是什么"、"XX怎么样"、"XX的特点"等需要理解的问题
            - 需要基于文档内容进行推理和解释的问题
            - 不适合：简单的关键词查找（应使用search_documents）
            
            Args:
                question: 关于文档内容的问题，需要理解和解释
            
            Returns:
                基于文档内容的详细回答
            """
            tool_name = "query_knowledge_base"
            self._mw_tool_start(tool_name, {"question": question})
            try:
                # 调用RAG链（使用简化输入，不包含历史记录）
                # 注意：Agent模式下，历史记录由Checkpointer管理
                result = rag_chain.invoke({"input": question, "history": []})
                self._mw_tool_end(tool_name, result)
                return result
            except Exception as e:
                logger.error(f"RAG查询失败: {e}")
                self._mw_tool_error(tool_name, e)
                return f"查询知识库失败: {str(e)}"

        return query_knowledge_base
    
    def _create_search_tool(self):
        """创建文档关键词搜索工具"""
        @tool
        def search_documents(keyword: str) -> str:
            """在文档中精确搜索关键词，返回包含该关键词的原始段落（会高亮显示关键词）。
            
            使用场景：
            - 用户明确说"搜索XX"、"查找XX"、"找XX这个词"
            - 需要看到关键词在文档中的原始出现位置
            - 需要精确匹配，而不是理解性回答
            
            注意：如果用户问"XX是什么"、"XX怎么样"，应使用query_knowledge_base
            
            Args:
                keyword: 要搜索的关键词或短语（如"崔老道"、"董妃坟"）
            
            Returns:
                包含关键词的文档段落列表，关键词会被【】高亮标记
            """
            tool_name = "search_documents"
            self._mw_tool_start(tool_name, {"keyword": keyword})
            try:
                keyword = (keyword or "").strip().strip('\'"“”‘’')
                if not keyword:
                    return "关键词不能为空"

                # 教学版：直接使用 retriever（向量相似度召回），逻辑更直观
                docs = self.retriever.invoke(keyword)
                if not docs:
                    out = f"未找到包含关键词 '{keyword}' 的文档内容。"
                    self._mw_tool_end(tool_name, out)
                    return out

                results = []
                for i, doc in enumerate(docs[:5], 1):
                    source = doc.metadata.get("source", "未知来源")
                    content = (doc.page_content or "").strip()
                    # 高亮关键词（不区分大小写）
                    highlighted_content = re.sub(
                        re.escape(keyword), 
                        f"【{keyword}】", 
                        content, 
                        flags=re.IGNORECASE
                    )
                    results.append(f"[结果{i} - {source}]:\n{highlighted_content}")

                out = f"找到 {len(docs)} 个相关结果（显示前5个）：\n\n" + "\n\n".join(results)
                self._mw_tool_end(tool_name, out)
                return out
            except Exception as e:
                logger.error(f"文档搜索失败: {e}")
                self._mw_tool_error(tool_name, e)
                return f"搜索文档失败: {str(e)}"
                
        return search_documents
    
    def _create_summary_tool(self):
        """创建文档摘要工具"""
        @tool
        def summarize_document(topic: str) -> str:
            """生成指定主题的文档摘要。
            
            当用户需要了解某个主题的概览时，使用这个工具。
            
            Args:
                topic: 要摘要的主题或关键词
            
            Returns:
                该主题的文档摘要
            """
            tool_name = "summarize_document"
            self._mw_tool_start(tool_name, {"topic": topic})
            try:
                # 检索相关文档
                docs = self.retriever.invoke(topic)
                
                if not docs:
                    out = f"未找到关于主题 '{topic}' 的文档内容"
                    self._mw_tool_end(tool_name, out)
                    return out
                
                # 合并文档内容
                combined_content = "\n\n".join([
                    doc.page_content.strip() 
                    for doc in docs[:3]  # 最多使用3个文档
                ])
                
                # 使用LLM生成摘要
                summary_prompt = f"""请为以下文档内容生成一个简洁的摘要，主题是：{topic}

文档内容：
{combined_content}

请生成一个200字左右的摘要："""
                
                response = self.llm.invoke(summary_prompt)
                summary = response.content if hasattr(response, 'content') else str(response)
                out = f"主题 '{topic}' 的摘要：\n{summary}"
                self._mw_tool_end(tool_name, out)
                return out
            except Exception as e:
                logger.error(f"文档摘要失败: {e}")
                self._mw_tool_error(tool_name, e)
                return f"生成摘要失败: {str(e)}"

        return summarize_document
