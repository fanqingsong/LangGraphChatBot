
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from pydantic import BaseModel
from typing import Optional, List, Dict
import os
from dotenv import load_dotenv

load_dotenv(".env")



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
    },
    "zhipu": {
        "base_url": os.getenv("ZHIPU_API_URL", "https://api.siliconflow.cn/v1"),
        "api_key": os.getenv("ZHIPU_API_KEY", ""),
        "chat_model": os.getenv("ZHIPU_API_MODEL", 'Qwen/Qwen2.5-7B-Instruct'),
        "embedding_model": os.getenv("ZHIPU_API_EMBEDDING_MODEL"),
    }
}

DEFAULT_LLM_TYPE = "zhipu"
DEFAULT_TEMPERATURE = 0

config = MODEL_CONFIGS[DEFAULT_LLM_TYPE]

# llm = ChatOpenAI(model="gpt-3.5-turbo")
llm = ChatOpenAI(
    base_url=config["base_url"],
    api_key=config["api_key"],
    model=config["chat_model"],
    temperature=DEFAULT_TEMPERATURE,
    timeout=30,  # 添加超时配置（秒）
    max_retries=2  # 添加重试次数
)
# llm = ChatOpenAI(temperature=0)


if __name__ == "__main__":
    print(llm.invoke("你好"))
