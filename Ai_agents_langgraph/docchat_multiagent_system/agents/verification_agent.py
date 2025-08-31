from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, List
from langchain.schema import Document
from config.settings import settings
import logging
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
mistral_api_key = os.getenv("MISTRAL_API_KEY")

logger = logging.getLogger(__name__)

class VerificationAgent:
    def __init__(self):
        model_id = "qwen/qwen3-32b" # 	qwen/qwen3-32b  openai/gpt-oss-120b llama3-8b-8192 
    
        self.llm = ChatGroq(model=model_id,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            verbose=1)
        
        self.prompt = ChatPromptTemplate.from_template(
            """Verify the following answer against the provided context. Check for:
            1. Direct factual support (YES/NO)
            2. Unsupported claims (list)
            3. Contradictions (list)
            4. Relevance to the question (YES/NO)
            
            Respond in this format:
            Supported: YES/NO
            Unsupported Claims: [items]
            Contradictions: [items]
            Relevant: YES/NO
            
            Answer: {answer}
            Context: {context}
            """
        )
        
    def check(self, answer: str, documents: List[Document]) -> Dict:
        """Verify the answer against the provided documents."""
        context = "\n\n".join([doc.page_content for doc in documents])
        
        chain = self.prompt | self.llm | StrOutputParser()
        try:
            verification = chain.invoke({
                "answer": answer,
                "context": context
            })
            logger.info(f"Verification report: {verification}")
            logger.info(f"Context used: {context}")
        except Exception as e:
            logger.error(f"Error verifying answer: {e}")
            raise
        
        return {
            "verification_report": verification,
            "context_used": context
        }