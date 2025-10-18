import os
import json
import getpass
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, MessageGraph

from langchain_groq import ChatGroq
from langchain_mistralai import MistralAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
mistral_api_key = os.getenv("MISTRAL_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

model_id = "llama-3.1-8b-instant" # 	qwen/qwen3-32b  openai/gpt-oss-20b llama-3.1-8b-instant
    
llm = ChatGroq(model=model_id,
    temperature=0,
    max_tokens=100,
    timeout=None,
    max_retries=2,
    verbose=1)


text = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert assistant who provides concise and accurate answers.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
What is the capital of Canada?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
print(llm.invoke(text).content)