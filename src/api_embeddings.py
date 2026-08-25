import uuid
from datetime import datetime
from typing import List, Union

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from globals import use_openai_client

router = APIRouter()


class EmbeddingRequest(BaseModel):
    """嵌入请求模型"""
    input: Union[str, List[str]] = Field(
        ...,
        description="Text string or list of strings to embed",
    )
    model: str = Field(
        default="text-embedding-004",
        description="Model name to use for embedding",
    )

    @field_validator('input')
    @classmethod
    def validate_input(cls, v):
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("input cannot be empty")
            if len(v) > 8000:
                raise ValueError("input string exceeds maximum length of 8000")
        elif isinstance(v, list):
            if not v:
                raise ValueError("input list cannot be empty")
            if len(v) > 256:
                raise ValueError("batch size exceeds maximum limit of 256 strings")
            for item in v:
                if not isinstance(item, str) or not item.strip():
                    raise ValueError("all items in input list must be non-empty strings")
                if len(item) > 8000:
                    raise ValueError(f"input string exceeds maximum length: {len(item)} > 8000")
        else:
            raise ValueError("input must be a string or list of strings")
        return v


def get_request_id(request: Request) -> str:
    """从请求头获取或生成请求ID"""
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = str(uuid.uuid4())
    return request_id


def _sync_create_embeddings(client, input_data, model_name):
    """在工作线程池中执行远程 OpenAI 客户端调用"""
    return client.embeddings.create(
        input=input_data,
        model=model_name,
    )


@router.post("/embeddings")
async def create_embeddings(
    request: Request,
    req: EmbeddingRequest = Body(...),
    request_id: str = Depends(get_request_id),
):
    """
    创建文本嵌入

    支持 OpenAI 兼容的 embeddings 接口。
    """
    log = logger.bind(request_id=request_id)
    ts_current = datetime.now()

    try:
        with use_openai_client() as client:
            response = await run_in_threadpool(
                _sync_create_embeddings,
                client=client,
                input_data=req.input,
                model_name=req.model,
            )

        elapsed = (datetime.now() - ts_current).total_seconds()
        dims = len(response.data[0].embedding) if response.data else 0
        log.info(
            f"✅ Embedding generated | "
            f"model={req.model} "
            f"| duration={elapsed:.3f}s "
            f"| dimensions={dims} "
            f"| request_id={request_id}"
        )

        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": item.embedding,
                    "index": idx,
                }
                for idx, item in enumerate(response.data)
            ],
            "model": req.model,
            "usage": {
                "prompt_tokens": getattr(response, "usage", None).prompt_tokens if hasattr(response, "usage") and response.usage else 0,
                "total_tokens": getattr(response, "usage", None).total_tokens if hasattr(response, "usage") and response.usage else 0,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"embedding failed | request_id={request_id}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "embedding_failed",
                "message": "Embedding request failed, please check server logs",
                "request_id": request_id,
            },
        )
