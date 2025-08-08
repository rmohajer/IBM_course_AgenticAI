#%%
"""Module for interfacing with IBM watsonx.ai LLMs."""

import logging
from typing import Dict, Any, Optional

from langchain_groq import ChatGroq
from llama_index.llms.langchain import LangChainLLM
from llama_index.embeddings.mistralai import MistralAIEmbedding
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods

import config

logger = logging.getLogger(__name__)

def create_watsonx_embedding() -> MistralAIEmbedding:
    """Creates an IBM Watsonx Embedding model for vector representation.
    
    Returns:
        WatsonxEmbeddings model.
    """
    watsonx_embedding = MistralAIEmbedding(
        model=config.EMBEDDING_MODEL_ID
    )
    
    logger.info(f"Created Watsonx Embedding model: {config.EMBEDDING_MODEL_ID}")
    return watsonx_embedding


def create_watsonx_llm(
    temperature: float = config.TEMPERATURE,
    max_new_tokens: int = config.MAX_NEW_TOKENS
) -> ChatGroq:
    """Creates an IBM Watsonx LLM for generating responses.
    
    Args:
        temperature: Temperature for controlling randomness in generation (0.0 to 1.0).
        max_new_tokens: Maximum number of new tokens to generate.
        decoding_method: Decoding method to use (sample, greedy).
        
    Returns:
        WatsonxLLM model.
    """

    
    llm = ChatGroq(model=config.LLM_MODEL_ID,
        temperature=temperature,
        max_tokens=max_new_tokens,
        timeout=None,
        max_retries=2,
        verbose=1)
    watson_llm = LangChainLLM(llm=llm)

    logger.info(f"Created LLM model: {config.LLM_MODEL_ID}")
    return watson_llm

def change_llm_model(new_model_id: str) -> None:
    """Change the LLM model to use.
    
    Args:
        new_model_id: New LLM model ID to use.
    """
    global config
    config.LLM_MODEL_ID = new_model_id
    logger.info(f"Changed LLM model to: {new_model_id}")
# %%
