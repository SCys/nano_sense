from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger

from api_audio_transcriptions import router as audio_router
from api_audio_synthesis import router as synthesis_router
from api_embeddings import router as embeddings_router
from api_vision_detection import router as vision_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server starting...")
    yield
    logger.info("Server shutting down...")


app = FastAPI(title="NanoSense API", description="Lightweight Multi-Modal AI Inference Gateway", version="0.1.0", lifespan=lifespan)

# 注册路由
app.include_router(audio_router, prefix="/v1/audio")
app.include_router(synthesis_router, prefix="/v1/audio")
app.include_router(embeddings_router, prefix="/v1")
app.include_router(vision_router, prefix="/v1/vision")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
