from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_community.chat_models.sambanova import ChatSambaNovaCloud

from struct_vs_unstruct.helpers.logger import logger


MODEL_ID = "llama3-405b"

logger.info("Using %s for inference", MODEL_ID)

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.03,  # ~ 10 requests per minute
    check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=10,  # Controls the maximum burst size.
)

model_kwargs = {
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 15,
    "max_tokens": 1800
}

model = ChatSambaNovaCloud(
    model=MODEL_ID,
    rate_limiter=rate_limiter,
    **model_kwargs
)