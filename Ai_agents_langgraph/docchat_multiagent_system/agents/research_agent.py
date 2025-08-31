from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, List
from langchain.schema import Document
from config.settings import settings
import logging
import os
import json
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_mistralai import MistralAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
mistral_api_key = os.getenv("MISTRAL_API_KEY")


logger = logging.getLogger(__name__)

class ResearchAgent:
    def __init__(self):
        """Initialize the research agent with the OpenAI model."""
        model_id = "qwen/qwen3-32b" # 	qwen/qwen3-32b  openai/gpt-oss-120b llama3-8b-8192 
    
        self.llm = ChatGroq(model=model_id,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            verbose=1)
        
        self.prompt = ChatPromptTemplate.from_template(
            """Answer the following question based on the provided context. Be precise and factual.
            
            Question: {question}
            
            Context:
            {context}
            
            If the context is insufficient, respond with: "I cannot answer this question based on the provided documents."
            """
        )
        
    def generate(self, question: str, documents: List[Document]) -> Dict:
        """Generate an initial answer using the provided documents."""
        context = "\n\n".join([doc.page_content for doc in documents])
        
        chain = self.prompt | self.llm | StrOutputParser()
        try:
            answer = chain.invoke({
                "question": question,
                "context": context
            })
            logger.info(f"Generated answer: {answer}")
            logger.info(f"Context used: {context}")
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
        
        return {
            "draft_answer": answer,
            "context_used": context
        }