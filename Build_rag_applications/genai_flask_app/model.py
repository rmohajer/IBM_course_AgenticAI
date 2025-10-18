import os
import json
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.output_parsers import JsonOutputParser

from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

model_id = "llama-3.1-8b-instant" # 	qwen/qwen3-32b  openai/gpt-oss-20b llama-3.1-8b-instant
    
llm = ChatGroq(model=model_id,
    temperature=0,
    max_tokens=100,
    timeout=None,
    max_retries=2,
    verbose=1)

# Define JSON output structure
class AIResponse(BaseModel):
    summary: str = Field(description="Summary of the user's message")
    sentiment: int = Field(description="Sentiment score from 0 (negative) to 100 (positive)")
    category: str = Field(description="Category of the inquiry (e.g., billing, technical, general)")
    action: str = Field(description="Recommended action for the support rep")

# JSON output parser
json_parser = JsonOutputParser(pydantic_object=AIResponse)
format_instructions = json_parser.get_format_instructions()


prompt_template = PromptTemplate(
    template='''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{format_instructions}

{user_prompt}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
''',
    input_variables=["system_prompt", "user_prompt"],
    partial_variables={"format_instructions": format_instructions},
)


def get_ai_response(model, template, system_prompt, user_prompt):
    chain = template | model | json_parser
    return chain.invoke({'system_prompt':system_prompt, 'user_prompt':user_prompt, 'format_prompt':json_parser.get_format_instructions()})


def model_response(system_prompt, user_prompt):
    return get_ai_response(llm, prompt_template, system_prompt, user_prompt)





