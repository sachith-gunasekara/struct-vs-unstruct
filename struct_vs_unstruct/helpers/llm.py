from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_community.chat_models.sambanova import ChatSambaNovaCloud
from langchain_mistralai import ChatMistralAI

from struct_vs_unstruct.helpers.logger import logger
from struct_vs_unstruct.helpers.config import read_config

config = read_config()

llama = True if "llama" in config["MODE"]["model"] else False

MODEL_ID = "llama3-405b" if llama else "mistral-large-2407" 

logger.info("Using %s for inference", MODEL_ID)

rate_limiter = InMemoryRateLimiter(
    requests_per_second=1,  # ~ 10 requests per minute
    check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=1,  # Controls the maximum burst size.
)

model_kwargs = {
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 15,
    "max_tokens": 2048 if llama else 10240
}


if llama:
    model = ChatSambaNovaCloud(
        model=MODEL_ID,
        rate_limiter=rate_limiter,
        **model_kwargs
    )
else:
    model = ChatMistralAI(
        model=MODEL_ID,
        rate_limiter=rate_limiter,
        **model_kwargs
    )