#%% Import the necessary packages

import groq
from groq import Groq
from langchain_groq import ChatGroq
from langchain_mistralai import MistralAIEmbeddings
import os
from dotenv import load_dotenv

import gradio as gr

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
mistral_api_key = os.getenv("MISTRAL_API_KEY")

#%% Specify the model and project settings 
# (make sure the model you wish to use is commented out, and other models are commented)
#model_id = 'mistralai/mixtral-8x7b-instruct-v01' # Specify the Mixtral 8x7B model
model_id = "llama3-8b-8192" # Specify IBM's Granite 3.3 8B model

llm = ChatGroq(model=model_id,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        verbose=1)

#%%
# Function to generate a response from the model
def generate_response(prompt_txt):
    generated_response = llm.invoke(prompt_txt).content
    return generated_response
# Create Gradio interface
chat_application = gr.Interface(
    fn=generate_response,
    allow_flagging="never",
    inputs=gr.Textbox(label="Input", lines=2, placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Output"),
    title="Watsonx.ai Chatbot",
    description="Ask any question and the chatbot will try to answer."
)
# Launch the app
chat_application.launch(server_name="127.0.0.1", server_port= 7860, share=True)