import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from modules.llm_mistral import query_mistral  # Your llama-cpp inference function

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mistral_api")

app = FastAPI(title="Mistral-7B Inference API", version="1.0")

class QueryRequest(BaseModel):
    user_input: str
    max_new_tokens: int = 200

class QueryResponse(BaseModel):
    response: str
    latency_sec: float
    tokens_used: int

@app.post("/generate", response_model=QueryResponse)
async def generate_text(query: QueryRequest):
    user_input = query.user_input.strip()
    max_tokens = query.max_new_tokens

    if not user_input:
        raise HTTPException(status_code=400, detail="Empty input")

    try:
        start = time.time()
        response, latency, tokens = query_mistral(user_input, max_new_tokens=max_tokens)
        elapsed = time.time() - start

        logger.info(f"Request processed: tokens={tokens}, latency={elapsed:.2f}s")
        return QueryResponse(response=response, latency_sec=elapsed, tokens_used=tokens)

    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

