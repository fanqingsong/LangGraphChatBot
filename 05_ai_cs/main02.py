from typing import List, Dict, Optional
from langgraph.graph import Graph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from sqlalchemy import create_engine, Column, String, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import json
import sqlite3, os
from langgraph.graph import StateGraph, START, END, MessagesState
import logging


from dotenv import load_dotenv

# Author:@南哥AGI研习社 (B站 or YouTube 搜索“南哥AGI研习社”)


# 加载.env文件中的环境变量
# .env在上一级目录中，请修改路径
load_dotenv('.env')


# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ================ 数据库配置 ================
Base = declarative_base()

class UserRecord(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    email = Column(String(100))
    extra_data = Column(JSON)  # 存储其他字段

engine = create_engine("sqlite:///user_data.db")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# ================ 数据模型 ================
class UserInfo(BaseModel):
    name: str = Field(description="用户全名")
    email: str = Field(description="有效邮箱地址")
    age: Optional[int] = Field(None, description="用户年龄")
    preferences: List[str] = Field(default_factory=list, description="兴趣列表")


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
        "api_key": os.getenv("SILICONFLOW_API_KEY", "kkk"),
        "chat_model": os.getenv("SILICONFLOW_API_MODEL", 'Qwen/Qwen2.5-7B-Instruct'),
        "embedding_model": os.getenv("SILICONFLOW_API_EMBEDDING_MODEL"),
    }
}

DEFAULT_LLM_TYPE = "siliconflow"
DEFAULT_TEMPERATURE = 0.7

config = MODEL_CONFIGS[DEFAULT_LLM_TYPE]
if not config["api_key"]:
    raise ValueError("API key is not set for the selected LLM type.")

# llm = ChatOpenAI(model="gpt-3.5-turbo")
llm = ChatOpenAI(
    base_url=config["base_url"],
    api_key=config["api_key"],
    model=config["chat_model"],
    temperature=DEFAULT_TEMPERATURE,
    timeout=30,  # 添加超时配置（秒）
    max_retries=2  # 添加重试次数
)


system_prompt = """你是一个专业的信息收集助手，请逐步获取以下信息：
{fields}

规则：
1. 每次只问1个问题
2. 用JSON格式返回已收集的数据"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=system_prompt.format(
        fields="\n".join(f"- {k}: {v['description']}" 
               for k,v in UserInfo.schema()["properties"].items())
    )),
    ("human", "{input}"),
    ("placeholder", "{messages}"),
])

# ================ 工具定义 ================
def save_to_db(data: dict):
    with Session() as session:
        user = UserRecord(
            name=data["name"],
            email=data["email"],
            extra_data={
                "age": data.get("age"),
                "preferences": data.get("preferences", [])
            }
        )
        session.add(user)
        session.commit()
    return f"用户 {data['name']} 已保存"

# ================ LangGraph工作流 ================
def should_continue(state: List[dict]):
    last_msg = state["messages"][-1]
    try:
        data = json.loads(last_msg.content)
        return "end" if all(v for k,v in data.items() if k != "preferences") else "continue"
    except:
        return "continue"

workflow = StateGraph(MessagesState)

workflow.add_node("collect", lambda state: {
    "messages": [llm.invoke(prompt.format(
        input=state.get("input", ""),
        messages=state["messages"]
    ))]
})

workflow.add_node("save", lambda state: {
    "result": save_to_db(json.loads(state["messages"][-1].content))
})

workflow.add_conditional_edges(
    "collect",
    should_continue,
    {"continue": "collect", "end": "save"}
)

workflow.set_entry_point("collect")
workflow.set_finish_point("save")

app = workflow.compile()

# 将构建的graph可视化保存为 PNG 文件
def save_graph_visualization(graph: StateGraph, filename: str = "graph.png") -> None:
    try:
        with open(filename, "wb") as f:
            f.write(graph.get_graph().draw_mermaid_png())
        logger.info(f"Graph visualization saved as {filename}")
    except IOError as e:
        logger.info(f"Warning: Failed to save graph visualization: {str(e)}")

# 保存 Graph 可视化图
# save_graph_visualization(app)


# ================ 测试运行 ================
if __name__ == "__main__":
    # 初始化对话
    messages = [AIMessage(content="{}")]  # 初始空JSON
    
    while True:
        user_input = input("用户: ")
        if user_input.lower() == "exit":
            break
            
        result = app.invoke({
            "input": user_input,
            "messages": messages
        })
        
        if "result" in result:
            print("系统:", result["result"])
            break
        else:
            messages = result["messages"]
            print("AI:", messages[-1].content)

