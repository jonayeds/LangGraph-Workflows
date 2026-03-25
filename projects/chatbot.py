from  langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


model = ChatGroq(model='llama-3.3-70b-versatile')
def chat_node(state:ChatState):
    messages=state['messages']
    response=model.invoke(messages)
    return {'messages':response}


conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer=SqliteSaver(conn=conn)
graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot=graph.compile(checkpointer=checkpointer)



thread_id='1'
config={'configurable':{'thread_id':thread_id}}
chat_history = list()
for checkpoint_tuple in checkpointer.list(config=config):
    channel_values = checkpoint_tuple.checkpoint.get('channel_values', {})
    messages = channel_values.get('messages',[])
    for msg in messages:
        chat_history.insert(0,msg)

for msg in chat_history:
    role = msg.__class__.__name__
    print(f'{"You" if role=="HumanMessage" else "Ai"}: {msg.content}')
while True:
    user_message = input("Type here: ")
    if user_message.strip().lower() in ['exit', 'quit', 'bye']:
        break

    print("AI: ", end="", flush=True)
    for message_chunk,metadata in chatbot.stream({'messages':HumanMessage(content=user_message)}, config=config, stream_mode='messages'):
        if message_chunk.content:
            print(message_chunk.content, end=" ", flush=True)

    print()

