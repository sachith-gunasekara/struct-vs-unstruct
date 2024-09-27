from langchain_nvidia_ai_endpoints import ChatNVIDIA

MODEL_ID = "meta/llama-3.1-405b-instruct"

model_kwargs = {
    "temperature": 0.2,
    "top_p": 0.9,
    "max_tokens": 1024
}

model = ChatNVIDIA(
    model=MODEL_ID,
    **model_kwargs
)