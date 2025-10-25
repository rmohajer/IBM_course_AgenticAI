
#%%
from langchain.chains import LLMChain  # For creating chains of operations with LLMs
from langchain.prompts import PromptTemplate  # For defining prompt templates
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
model_id = "qwen/qwen3-32b" # 	qwen/qwen3-32b  openai/gpt-oss-120b llama3-8b-8192
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model=model_id,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            verbose=1)


response = llm.invoke("How to read a book effectively?")
print(response)
# %%
def remove_non_ascii(text):
    return ''.join(i for i in text if ord(i) < 128)

def product_assistant(ascii_transcript):
    system_prompt = """You are an intelligent assistant specializing in financial products;
    your task is to process transcripts of earnings calls, ensuring that all references to
     financial products and common financial terms are in the correct format. For each
     financial product or common term that is typically abbreviated as an acronym, the full term 
    should be spelled out followed by the acronym in parentheses. For example, '401k' should be
     transformed to '401(k) retirement savings plan', 'HSA' should be transformed to 'Health Savings Account (HSA)' , 'ROA' should be transformed to 'Return on Assets (ROA)', 'VaR' should be transformed to 'Value at Risk (VaR)', and 'PB' should be transformed to 'Price to Book (PB) ratio'. Similarly, transform spoken numbers representing financial products into their numeric representations, followed by the full name of the product in parentheses. For instance, 'five two nine' to '529 (Education Savings Plan)' and 'four zero one k' to '401(k) (Retirement Savings Plan)'. However, be aware that some acronyms can have different meanings based on the context (e.g., 'LTV' can stand for 'Loan to Value' or 'Lifetime Value'). You will need to discern from the context which term is being referred to  and apply the appropriate transformation. In cases where numerical figures or metrics are spelled out but do not represent specific financial products (like 'twenty three percent'), these should be left as is. Your role is to analyze and adjust financial product terminology in the text. Once you've done that, produce the adjusted transcript and a list of the words you've changed"""

    # Concatenate the system prompt and the user transcript
    prompt_input = system_prompt + "\n" + ascii_transcript

    # Create a messages object
    messages = [
        {
            "role": "user",
            "content": prompt_input
        }
    ]

    
    
    # Send the input messages to the model and retrieve its response
    response = llm.invoke(input=messages)
    
    # Extract and return the content of the model's first response choice
    return response.content

# Define the prompt template
template = """
Generate meeting minutes and a list of tasks based on the provided context.

Context:
{context}

Meeting Minutes:
- Key points discussed
- Decisions made

Task List:
- Actionable items with assignees and deadlines
"""
prompt = ChatPromptTemplate.from_template(template)

# Define the chain
chain = (
    {"context": RunnablePassthrough()}  # Pass the transcript as context
    | prompt
    | llm
    | StrOutputParser()
)

chain.invoke({"context": "Discuss project timeline and assign tasks to team members."})
# %%
