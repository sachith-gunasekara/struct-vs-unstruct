import os

from langchain_nvidia_ai_endpoints import ChatNVIDIA

from struct_vs_unstruct.helpers.logger import logger

MODEL_ID = "meta/llama-3.1-405b-instruct"

logger.info("Using %s for inference", MODEL_ID)

model_kwargs = {
    "temperature": 0.2,
    "top_p": 0.9,
    "max_tokens": 4096
}

model = ChatNVIDIA(
    model=MODEL_ID,
    **model_kwargs
)