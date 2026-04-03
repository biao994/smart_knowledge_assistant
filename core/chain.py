import logging

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI



logger = logging.getLogger(__name__)

def format_docs(docs) -> str:
    """
    格式化检索结果
    
    Args：
        docs: 检索到的文档列表
    
    Returns:
        格式化后的文档字符串
    """
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "未知来源")
        content = doc.page_content.strip()
        formatted.append(f"[文档{i} - {source}]:\n{content}")
    return "\n\n".join(formatted)
    

def build_prompt_template() -> ChatPromptTemplate:
    """
    构建提示模板

    注意： RunnableWithMessageHistory会自动将历史消息注入到history变量中
    我们需要在prompt中同时包含context和history

    Returns:
        ChatPromptTemplate对象
    """
    return ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的智能助手，基于提供的上下文和对话历史回答用户问题。

## 上下文信息：
{context}

## 回答要求：
1. 严格基于上下文信息回答，不要编造不知道的内容
2. 如果上下文信息不足，请明确告知并建议用户提供更多信息
3. 回答要专业、准确、简洁
4. 如果问题与知识库无关，请礼貌拒绝
5. 结合对话历史理解用户意图，保持对话连贯性

请回答："""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

def build_rag_chain(retriever: BaseRetriever,config:dict) -> Runnable:
    """
    构建RAG链(基础链，将被RunnableWithMessageHistory包装)

    注意：RunnableWithMessageHistory期望链接受字典输入（包含input键），
    然后它会自动将历史消息注入到字典的history键中。
    我们需要在链内部处理context的获取。

    Args:
        retriever: 文档检索器
        config: 配置字典

    Returns:
        Runnable对象，包含RAG链

    """
    
    # 构建提示模板
    prompt = build_prompt_template()

    # 初始化LLM
    llm = ChatOpenAI(
        model=config.get("model", "gpt-3.5-turbo"),
        temperature=config.get("temperature", 0.1),
        max_retries=config.get("llm_max_retries", 3),
        timeout=config.get("llm_timeout", 30),
        max_tokens=config.get("max_tokens", 1000)
    )

    #  提取问题和context
    def prepare_input(user_input_dict):
        """
        准备输入：从用户输入字典中提取问题，并获取context

        RunnableWithMessageHistory会传入字典，格式为：
        {
            "input": "用户问题",
            "history": [...]  # 历史消息列表（自动注入）
        }

        我们需要：
        1. 提取问题（从input键）
        2. 通过retriever获取context
        3. 返回包含context、input和history的字典      
        """

        # 提取问题
        question = user_input_dict.get("input", "")
        history = user_input_dict.get("history", [])

        # 获取context
        docs = retriever.invoke(question)
        context = format_docs(docs)

        return {
            "context": context,
            "input": question,
            "history": history
        }
    
    # 构建链： 接受字典输入 -> 准备context和input -> prompt -> llm -> 输出
    rag_chain = (
        RunnableLambda(prepare_input)
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info("RAG链构建完成")
    return rag_chain
