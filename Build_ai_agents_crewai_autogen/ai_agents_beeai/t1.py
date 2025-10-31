#%% install libraries

%uv pip install 'beeai-framework[wikipedia]==0.1.35' 

# %% 

import os

from langchain_groq import ChatGroq
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

model_id = "llama-3.1-8b-instant" # 	qwen/qwen3-32b  openai/gpt-oss-20b llama-3.1-8b-instant
    
llm = ChatGroq(model=model_id,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    verbose=1)

print("Environment configured successfully!")

# %%
import asyncio
import logging
from beeai_framework.backend import ChatModel, ChatModelParameters, UserMessage, SystemMessage
# Initialize the chat model
async def basic_chat_example():
    # Create a chat model instance (works with OpenAI, WatsonX, etc.)
    
    # Create a conversation about something everyone finds interesting
    messages = [
        SystemMessage(content="You are a helpful AI assistant and creative writing expert."),
        UserMessage(content="Help me brainstorm a unique business idea for a food delivery service that doesn't exist yet.")
    ]

    llm = ChatModel.from_name(f"groq/{model_id}", ChatModelParameters(temperature=0))
    
    # Generate response using create() method
    response = await llm.invoke(input=messages)
    
    print("User: Help me brainstorm a unique business idea for a food delivery service that doesn't exist yet.")
    print(f"Assistant: {response.get_text_content()}")
    
    return response

# %%
# Run the basic chat example
async def main() -> None:
    logging.getLogger('asyncio').setLevel(logging.CRITICAL) # Suppress unwanted warnings
    response = await basic_chat_example()

if __name__ == "__main__":
    import nest_asyncio
    import asyncio

    nest_asyncio.apply()  # allows nested event loops
    asyncio.get_event_loop().run_until_complete(main())
# %%
