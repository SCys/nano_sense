from datetime import datetime
from fastapi import APIRouter, Request
from loguru import logger
from globals import get_openai_client

router = APIRouter()


@router.post("/embeddings")
async def create_embeddings(request: Request):
    """
    curl "https://generativelanguage.googleapis.com/v1beta/openai/embeddings" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer GEMINI_API_KEY" \
    -d '{
        "input": "Your text string goes here",
        "model": "text-embedding-004"
    }'
    """
    data = await request.json()
    input_text = data.get("input")
    model = data.get("model", "text-embedding-004")

    if not input_text:
        return {"error": "input is required"}

    ts_current = datetime.now()
    try:
        client = get_openai_client()  # 懒加载
        response = client.embeddings.create(
            input=input_text,
            model=model,
        )

        logger.info(f"embedding time: {datetime.now() - ts_current}")

        return {
            "object": "embedding",
            "embedding": response.data[0].embedding,
            "index": 0,
        }
    except Exception as e:
        logger.exception("embedding failed")
        return {"error": str(e)}
