import asyncio
import json
import threading
import gradio as gr
from langchain_experimental.autonomous_agents import AutoGPT
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.tools import ReadFileTool, WriteFileTool
from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import Tool
import time


# OpenAI Embedding 模型
embeddings_model = OpenAIEmbeddings()

# OpenAI Embedding 向量维数
embedding_size = 1536
# 使用 Faiss 的 IndexFlatL2 索引
index = faiss.IndexFlatL2(embedding_size)
# 实例化 Faiss 向量数据库
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

# 构造 AutoGPT 的工具集
search = SerpAPIWrapper()
tools = [
    Tool(
        name="search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
    WriteFileTool(),
    ReadFileTool(),
]


def run_agent(task,history):

    chat_history = ChatMessageHistory()

    agent = AutoGPT.from_llm_and_tools(
        ai_name="Jarvis",
        ai_role="Assistant",
        tools=tools,
        llm=ChatOpenAI( model="gpt-4o", temperature=0), # base_url ="https://api.deepseek.com/v1",
        memory=vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.8}),  # 实例化 Faiss 的 VectorStoreRetriever
        chat_history_memory=chat_history,
    )

    thread = threading.Thread(target=agent.run,args=([task],), daemon=True)
    thread.start()

    read_message_count = 0
    all_messages = ""
    while True:

        message_history  = asyncio.run(chat_history.aget_messages())
        total_message_count = len(message_history)
        if total_message_count == read_message_count:
            print('waiting new messages...')
            time.sleep(1)
            continue
        new_message = message_history[read_message_count]
        read_message_count += 1

        new_message.pretty_print()
        if new_message.type =='human':
            continue
        all_messages += format_to_markdown(json.dumps({"type": new_message.type, "content": new_message.content.replace("\n", "")}))
        try:
            new_message_json = json.loads(new_message.content)
            if 'command' in new_message_json and 'name' in new_message_json['command'] and new_message_json['command']['name'] == 'finish':
                print("find finish message")
                break
        except json.decoder.JSONDecodeError:
            print('new_message is not JSON')

        time.sleep(1)
        yield all_messages
    yield all_messages


def format_to_markdown(json_str):
    """将 JSON 字符串转换为 Markdown 格式。如果输入不是 JSON，保持原样式返回。"""
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print('解析失败，返回原始字符串')
        print(e)
        return json_str

    def format_dict(d, title_level=2):
        """递归格式化字典为 Markdown"""
        md = ""

        for key, value in d.items():
            if isinstance(value, dict):
                md += f"{'#' * title_level} **{key}**:  \n\n{format_dict(value, title_level + 1)}"
            elif isinstance(value, list):
                md += f"{'#' * title_level} **{key}**:  \n\n" + "  \n\n".join(
                    [f"- {format_dict(v, title_level + 2) if isinstance(v, dict) else v}" for v in
                     value]
                ) + "  \n\n"
            else:
                md += f"{'#' * title_level} **{key}**:  \n\n{value}  \n\n"
        return md
    result =  ''
    d = data
    if 'type' in d:
        result += f'\n ------\n\n# {d["type"]}\n'
    if 'content' in d:
        d = d['content']

        try :
            d = json.loads(d)
        except json.JSONDecodeError as e:
            print(e)
        result += (format_dict(d) if type(d) is dict else d+"\n")

    print(f"formate result:\n {result}")

    return result

iface = gr.ChatInterface(
    fn=run_agent,
    title="AUTOGPT",
    chatbot=gr.Chatbot(height=600),
    type="messages"
)
if __name__ == '__main__':
    iface.launch(server_name="0.0.0.0",server_port=8081)