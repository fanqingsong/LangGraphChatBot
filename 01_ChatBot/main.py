import os
import re
import uuid
import time
import json
import logging
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.prompts import PromptTemplate
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from llms import get_llm
from langgraph.checkpoint.memory import MemorySaver
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
# from openai.error import OpenAIError
from openai import OpenAIError


# Author:@南哥AGI研习社 (B站 or YouTube 搜索“南哥AGI研习社”)


# 设置LangSmith环境变量 进行应用跟踪，实时了解应用中的每一步发生了什么
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = "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"


# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# prompt模版设置相关 根据自己的实际业务进行调整
PROMPT_TEMPLATE_TXT_SYS = "prompt_template_system.txt"
PROMPT_TEMPLATE_TXT_USER = "prompt_template_user.txt"

# openai:调用gpt模型,oneapi:调用oneapi方案支持的模型,ollama:调用本地开源大模型,qwen:调用阿里通义千问大模型
llm_type = "siliconflow"

# API服务设置相关
PORT = 8012

# 申明全局变量 全局调用
graph = None


# 定义消息类，用于封装API接口返回数据
# 定义Message类
class Message(BaseModel):
    role: str
    content: str

# 定义ChatCompletionRequest类
class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    stream: Optional[bool] = False
    userId: Optional[str] = None
    conversationId: Optional[str] = None

# 定义ChatCompletionResponseChoice类
class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None

# 定义ChatCompletionResponse类
class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[ChatCompletionResponseChoice]
    system_fingerprint: Optional[str] = None


# 定义chatbot的状态
class State(TypedDict):
    messages: Annotated[list, add_messages]


# 创建和配置chatbot的状态图
def create_graph(llm) -> StateGraph:
    try:
        # 构建graph
        graph_builder = StateGraph(State)

        # 定义chatbot的node
        def chatbot(state: State) -> dict:
            # 处理当前状态并返回 LLM 响应
            return {"messages": [llm.invoke(state["messages"])]}

        # 配置graph
        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)

        # 这里使用内存存储 也可以持久化到数据库
        memory = MemorySaver()

        # 编译生成graph并返回
        return graph_builder.compile(checkpointer=memory)

    except Exception as e:
        raise RuntimeError(f"Failed to create graph: {str(e)}")


# 将构建的graph可视化保存为 PNG 文件
def save_graph_visualization(graph: StateGraph, filename: str = "graph.png") -> None:
    try:
        with open(filename, "wb") as f:
            f.write(graph.get_graph().draw_mermaid_png())
        logger.info(f"Graph visualization saved as {filename}")
    except IOError as e:
        logger.info(f"Warning: Failed to save graph visualization: {str(e)}")


# 格式化响应，对输入的文本进行段落分隔、添加适当的换行符，以及在代码块中增加标记，以便生成更具可读性的输出
def format_response(response):
    # 使用正则表达式 \n{2, }将输入的response按照两个或更多的连续换行符进行分割。这样可以将文本分割成多个段落，每个段落由连续的非空行组成
    paragraphs = re.split(r'\n{2,}', response)
    # 空列表，用于存储格式化后的段落
    formatted_paragraphs = []
    # 遍历每个段落进行处理
    for para in paragraphs:
        # 检查段落中是否包含代码块标记
        if '```' in para:
            # 将段落按照```分割成多个部分，代码块和普通文本交替出现
            parts = para.split('```')
            for i, part in enumerate(parts):
                # 检查当前部分的索引是否为奇数，奇数部分代表代码块
                if i % 2 == 1:  # 这是代码块
                    # 将代码块部分用换行符和```包围，并去除多余的空白字符
                    parts[i] = f"\n```\n{part.strip()}\n```\n"
            # 将分割后的部分重新组合成一个字符串
            para = ''.join(parts)
        else:
            # 否则，将句子中的句点后面的空格替换为换行符，以便句子之间有明确的分隔
            para = para.replace('. ', '.\n')
        # 将格式化后的段落添加到formatted_paragraphs列表
        # strip()方法用于移除字符串开头和结尾的空白字符（包括空格、制表符 \t、换行符 \n等）
        formatted_paragraphs.append(para.strip())
    # 将所有格式化后的段落用两个换行符连接起来，以形成一个具有清晰段落分隔的文本
    return '\n\n'.join(formatted_paragraphs)


# 定义了一个异步函数lifespan，它接收一个FastAPI应用实例app作为参数。这个函数将管理应用的生命周期，包括启动和关闭时的操作
# 函数在应用启动时执行一些初始化操作，如加载上下文数据、以及初始化问题生成器
# 函数在应用关闭时执行一些清理操作
# @asynccontextmanager 装饰器用于创建一个异步上下文管理器，它允许你在 yield 之前和之后执行特定的代码块，分别表示启动和关闭时的操作
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    # 申明引用全局变量，在函数中被初始化，并在整个应用中使用
    global graph

    try:
        logger.info("正在初始化模型、定义Graph...")
        #（1）初始化LLM
        llm = get_llm(llm_type)
        #（2）定义Graph
        graph = create_graph(llm)
        #（3）将Graph可视化图保存
        save_graph_visualization(graph)
        logger.info("初始化完成")
    except Exception as e:
        logger.error(f"初始化过程中出错: {str(e)}")
        # raise 关键字重新抛出异常，以确保程序不会在错误状态下继续运行
        raise

    # yield 关键字将控制权交还给FastAPI框架，使应用开始运行
    # 分隔了启动和关闭的逻辑。在yield 之前的代码在应用启动时运行，yield 之后的代码在应用关闭时运行
    yield
    # 关闭时执行
    logger.info("正在关闭...")


# lifespan参数用于在应用程序生命周期的开始和结束时执行一些初始化或清理工作
app = FastAPI(lifespan=lifespan)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not graph:
        logger.error("服务未初始化")
        raise HTTPException(status_code=500, detail="服务未初始化")

    try:
        logger.info(f"收到聊天完成请求: {request}")

        query_prompt = request.messages[-1].content
        logger.info(f"用户问题是: {query_prompt}")

        config = {"configurable": {"thread_id": request.userId+"@@"+request.conversationId}}
        logger.info(f"用户当前会话信息: {config}")

        prompt_template_system = PromptTemplate.from_file(PROMPT_TEMPLATE_TXT_SYS)
        prompt_template_user = PromptTemplate.from_file(PROMPT_TEMPLATE_TXT_USER)
        prompt = [
            {"role": "system", "content": prompt_template_system.template},
            {"role": "user", "content": prompt_template_user.template.format(query=query_prompt)}
        ]

        if request.stream:
            return await handle_stream_response(prompt, config)
        else:
            return await handle_non_stream_response(prompt, config)

    except Exception as e:
        logger.error(f"处理聊天完成时出错:\n\n {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def handle_stream_response(prompt, config):
    async def generate_stream():
        chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
        try:
            async for message_chunk, metadata in graph.astream({"messages": prompt}, config, stream_mode="messages"):
                chunk = message_chunk.content
                logger.info(f"chunk: {chunk}")
                yield f"data: {json.dumps({'id': chunk_id,'object': 'chat.completion.chunk','created': int(time.time()),'choices': [{'index': 0,'delta': {'content': chunk},'finish_reason': None}]})}\n\n"
            yield f"data: {json.dumps({'id': chunk_id,'object': 'chat.completion.chunk','created': int(time.time()),'choices': [{'index': 0,'delta': {},'finish_reason': 'stop'}]})}\n\n"
        except OpenAIError as e:
            logger.error(f"OpenAI API 错误: {str(e)}")
            yield f"data: {json.dumps({'error': 'OpenAI API 错误'})}\n\n"
        except Exception as e:
            logger.error(f"流式响应处理错误: {str(e)}")
            yield f"data: {json.dumps({'error': '处理请求时发生错误'})}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")

async def handle_non_stream_response(prompt, config):
    max_retries = 3
    retry_delay = 1
    result = None

    for attempt in range(max_retries):
        try:
            events = graph.stream({"messages": prompt}, config)
            for event in events:
                for value in event.values():
                    result = value["messages"][-1].content
            break
        except OpenAIError as e:
            logger.warning(f"OpenAI API 错误 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error(f"OpenAI API 错误，已达到最大重试次数: {str(e)}")
                raise HTTPException(status_code=503, detail="服务暂时不可用，请稍后再试")
        except Exception as e:
            logger.error(f"非流式响应处理错误: {str(e)}")
            raise HTTPException(status_code=500, detail="处理请求时发生错误")

    if result is None:
        raise HTTPException(status_code=500, detail="无法获取有效响应")

    formatted_response = str(format_response(result))
    logger.info(f"格式化的搜索结果: {formatted_response}")

    response = ChatCompletionResponse(
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=Message(role="assistant", content=formatted_response),
                finish_reason="stop"
            )
        ]
    )
    logger.info(f"发送响应内容: \n{response}")
    return JSONResponse(content=response.model_dump())


if __name__ == "__main__":
    logger.info(f"在端口 {PORT} 上启动服务器")
    # uvicorn是一个用于运行ASGI应用的轻量级、超快速的ASGI服务器实现
    # 用于部署基于FastAPI框架的异步PythonWeb应用程序
    uvicorn.run(app, host="0.0.0.0", port=PORT)


