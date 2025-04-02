from typing import Dict, List, Annotated
from langgraph.graph import Graph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from pydantic import BaseModel, Field
import sqlite3, os
from langchain_core.tools import tool
# from langgraph.prebuilt import ToolNode, ToolExecutor
from langchain.tools import tool


# ================= 数据库部分 =================
DB_NAME = "user_info.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE,
        age INTEGER,
        preferences TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_to_db(data: dict):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO users (name, email, age, preferences)
    VALUES (?, ?, ?, ?)
    """, (data["name"], data["email"], data["age"], data["preferences"]))
    conn.commit()
    conn.close()

# ================= 数据模型 =================
class UserInfo(BaseModel):
    name: str = Field(description="用户全名")
    email: str = Field(description="有效的电子邮件地址")
    age: int = Field(description="用户年龄")
    preferences: List[str] = Field(description="用户兴趣偏好列表")

# ================= Agent 配置 =================

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
DEFAULT_TEMPERATURE = 0.7

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

system_prompt = """你是一个专业的信息收集助手，需要逐步获取以下用户信息：
{fields}

收集规则：
1. 每次只问1个明确的问题
2. 如果用户拒绝提供，记录为null
3. 获得全部信息后返回完整JSON"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=system_prompt.format(fields=UserInfo.schema()["properties"])),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# ================= 工具定义 =================
@tool
def store_user_info(info: Dict) -> str:
    """当收集完所有用户信息时调用此函数存储到数据库"""
    save_to_db(info)
    return f"用户 {info['name']} 的信息已存储"

# ================= LangGraph 工作流 =================
def should_continue(state: List[HumanMessage]):
    # 检查是否已收集完整信息
    collected = parse_collected_data(state[-1].content)
    return "continue" if None in collected.values() else "end"

def parse_collected_data(text: str) -> Dict:
    # 从对话历史解析已收集的数据（实际实现需更健壮）
    return {
        "name": "John" if "name is John" in text else None,
        # 其他字段解析...
    }

# tools = [store_user_info]
# tool_executor = ToolExecutor(tools)

# tool_node = ToolNode(tools=tools, tool_executor=tool_executor)
tool_node = ToolNode(tools=[store_user_info])

workflow = Graph()

workflow.add_node("collect_info", tool_node)  # 动态工具绑定

workflow.add_node("store_data", 
    ToolNode([store_user_info]))

workflow.add_conditional_edges(
    "collect_info",
    should_continue,
    {
        "continue": "collect_info",  # 继续收集
        "end": "store_data"         # 存储数据
    }
)

workflow.set_entry_point("collect_info")
workflow.set_finish_point("store_data")

app = workflow.compile()

# ================= 测试运行 =================
if __name__ == "__main__":
    init_db()
    
    # 模拟对话
    messages = []
    while True:
        user_input = input("User: ")
        messages.append(HumanMessage(content=user_input))
        result = app.invoke({"input": user_input, "messages": messages})
        
        if "需要更多信息" in result[-1].content:
            print("Agent:", result[-1].content)
        else:
            print("Agent: 信息收集完成！", result)
            break