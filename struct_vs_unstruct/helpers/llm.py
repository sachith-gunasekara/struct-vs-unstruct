import os

from langchain_openai import ChatOpenAI

MODEL_ID = "meta-llama/llama-3.1-405b-instruct:free"

model_kwargs = {
    "temperature": 0.2,
    "top_p": 0.9,
    "max_tokens": 1024
}

model = ChatOpenAI(
    model=MODEL_ID,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    **model_kwargs
)