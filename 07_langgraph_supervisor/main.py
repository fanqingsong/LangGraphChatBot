from langchain_openai import ChatOpenAI

from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
import os


from dotenv import load_dotenv

# Author:@南哥AGI研习社 (B站 or YouTube 搜索“南哥AGI研习社”)


# 加载.env文件中的环境变量
# .env在上一级目录中，请修改路径
load_dotenv('.env')


# model = ChatOpenAI(model="gpt-4o")


# 模型配置字典
MODEL_CONFIGS = {
    "openai": {
        "base_url": "https://nangeai.top/v1",
        "api_key": "sk-0OWbyfzUSwajhvqGoNbjIEEWchM15CchgJ5hIaN6qh9I3XRl",
        "chat_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small"

    },
    "oneapi": {
        "base_url": "http://139.224.72.218:3000/v1",
        "api_key": "sk-EDjbeeCYkD1OnI9E48018a018d2d4f44958798A261137591",
        "chat_model": "qwen-max",
        "embedding_model": "text-embedding-v1"
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "sk-80a72f794bc4488d85798d590e96db43",
        "chat_model": "qwen-max",
        "embedding_model": "text-embedding-v1"
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "chat_model": "deepseek-r1:14b",
        "embedding_model": "nomic-embed-text:latest"
    },
    "siliconflow": {
        "base_url": os.getenv("SILICONFLOW_API_URL", "https://api.siliconflow.cn/v1"),
        "api_key": os.getenv("SILICONFLOW_API_KEY", ""),
        "chat_model": os.getenv("SILICONFLOW_API_MODEL", 'Qwen/Qwen2.5-7B-Instruct'),
        "embedding_model": os.getenv("SILICONFLOW_API_EMBEDDING_MODEL"),
    }
}

DEFAULT_LLM_TYPE = "siliconflow"
DEFAULT_TEMPERATURE = 0

config = MODEL_CONFIGS[DEFAULT_LLM_TYPE]

# llm = ChatOpenAI(model="gpt-3.5-turbo")
model = ChatOpenAI(
    base_url=config["base_url"],
    api_key=config["api_key"],
    model=config["chat_model"],
    temperature=DEFAULT_TEMPERATURE,
    timeout=30,  # 添加超时配置（秒）
    max_retries=2  # 添加重试次数
)
# llm = ChatOpenAI(temperature=0)


# Create specialized agents

def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def web_search(query: str) -> str:
    """Search the web for information."""
    return (
        "Here are the headcounts for each of the FAANG companies in 2024:\n"
        "1. **Facebook (Meta)**: 67,317 employees.\n"
        "2. **Apple**: 164,000 employees.\n"
        "3. **Amazon**: 1,551,000 employees.\n"
        "4. **Netflix**: 14,000 employees.\n"
        "5. **Google (Alphabet)**: 181,269 employees."
    )

math_agent = create_react_agent(
    model=model,
    tools=[add, multiply],
    name="math_expert",
    prompt="You are a math expert. Always use one tool at a time."
)

research_agent = create_react_agent(
    model=model,
    tools=[web_search],
    name="research_expert",
    prompt="You are a world class researcher with access to web search. Do not do any math."
)

# Create supervisor workflow
workflow = create_supervisor(
    [research_agent, math_agent],
    # output_mode="last_message",
    model=model,
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, use research_agent. "
        "For math problems, use math_agent."
    )
)

app = workflow.compile()
#
# # Ensure the data passed to invoke is in the correct format
# input_data = {
#     "messages": [
#         {
#             "role": "user",
#             "content": "what's the combined headcount of the FAANG companies in 2024?"
#         }
#     ]
# }
#
# # Make sure the tool_calls data is in dictionary format
# # Here we assume that the problem might be related to the data passed to the tool_calls
# # If the data is generated dynamically, you need to ensure it's in the correct format
# # For simplicity, we assume there are no tool_calls here
# result = app.invoke(input_data)
#
# print(result)
#



# Ensure the data passed to invoke is in the correct format
input_data = {
    "messages": [
        {
            "role": "user",
            "content": "what's the sum of 2 and 3?"
        }
    ]
}

# Make sure the tool_calls data is in dictionary format
# Here we assume that the problem might be related to the data passed to the tool_calls
# If the data is generated dynamically, you need to ensure it's in the correct format
# For simplicity, we assume there are no tool_calls here
result = app.invoke(input_data)

print(result)

# 提取最终的AI消息
final_message = result["messages"][-1]
final_result = final_message.content

print(final_result)