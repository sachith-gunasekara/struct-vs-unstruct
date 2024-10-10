from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_community.chat_models.sambanova import ChatSambaNovaCloud

from struct_vs_unstruct.helpers.logger import logger


MODEL_ID = "meta/llama-3.1-405b-instruct"

logger.info("Using %s for inference", MODEL_ID)

model_kwargs = {
    "temperature": 0.2,
    "top_p": 0.9,
    "max_tokens": 4096
}

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.166,  # ~ 10 requests per minute
    check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=50,  # Controls the maximum burst size.
)

model = ChatSambaNovaCloud(
    model=MODEL_ID,
    rate_limiter=rate_limiter,
    **model_kwargs
)