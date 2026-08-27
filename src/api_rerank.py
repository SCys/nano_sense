import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from globals import use_rerank_model

router = APIRouter()

MAX_QUERY_LENGTH = 4000
MAX_DOCUMENTS = 512
MAX_DOC_LENGTH = 8000


class RerankRequest(BaseModel):
    """文本重排请求模型（对齐 Cohere / Jina / BAAI 标准格式）"""

    model: str = Field(
        default="bge-reranker-v2-m3",
        description="重排模型名称",
    )
    query: str = Field(
        ...,
        max_length=MAX_QUERY_LENGTH,
        description="检索查询词/问题",
    )
    documents: List[Union[str, Dict[str, Any]]] = Field(
        ...,
        description="候选文档列表（支持纯文本字符串或包含 text 字段的字典）",
    )
    top_n: Optional[int] = Field(
        default=None,
        ge=1,
        description="返回的相关性最高的文档数量，默认返回全部并按相关性排序",
    )
    return_documents: bool = Field(
        default=True,
        description="是否在响应中包含文档原始内容",
    )
    max_chunks_per_doc: Optional[int] = Field(
        default=None,
        description="兼容字段：每篇文档的最大切片数",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query cannot be empty")
        return v

    @field_validator("documents")
    @classmethod
    def validate_documents(cls, v: list) -> list:
        if not v:
            raise ValueError("documents list cannot be empty")
        if len(v) > MAX_DOCUMENTS:
            raise ValueError(f"documents count exceeds maximum limit of {MAX_DOCUMENTS}")
        return v


def get_request_id(request: Request) -> str:
    """从请求头获取或生成请求ID"""
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = str(uuid.uuid4())
    return request_id


def _extract_doc_text(doc: Union[str, Dict[str, Any]]) -> str:
    """从候选文档中提取纯文本"""
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        if "text" in doc:
            return str(doc["text"])
        if "content" in doc:
            return str(doc["content"])
        return str(doc)
    return str(doc)


def _sync_rerank_predict(
    tokenizer,
    model,
    query: str,
    doc_texts: List[str],
) -> Tuple[List[float], int]:
    """在工作线程池中执行重排推理与相关性打分"""
    pairs = [[query, text[:MAX_DOC_LENGTH]] for text in doc_texts]

    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    total_tokens = int(inputs["input_ids"].numel())

    # 将输入移至模型所在设备
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits.squeeze(-1)
        # 通过 Sigmoid 将得分归一化到 [0, 1] 之间
        probs = torch.sigmoid(logits)
        if probs.ndim == 0:
            probs = probs.unsqueeze(0)
        scores = probs.cpu().tolist()

    return scores, total_tokens


@router.post("/rerank")
async def rerank_documents(
    request: Request,
    req: RerankRequest = Body(...),
):
    """
    文档重排接口 (POST /v1/rerank)

    接收 query 与候选 documents 列表，使用 Cross-Encoder 计算相关度并降序返回。
    """
    request_id = get_request_id(request)
    log = logger.bind(request_id=request_id)

    doc_texts = [_extract_doc_text(d) for d in req.documents]

    try:
        with use_rerank_model() as (tokenizer, model):
            scores, total_tokens = await run_in_threadpool(
                _sync_rerank_predict,
                tokenizer=tokenizer,
                model=model,
                query=req.query,
                doc_texts=doc_texts,
            )

        # 按相关性得分降序排列
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # 截取 top_n
        if req.top_n is not None:
            indexed_scores = indexed_scores[: req.top_n]

        results = []
        for orig_idx, score in indexed_scores:
            item: Dict[str, Any] = {
                "index": orig_idx,
                "relevance_score": round(score, 6),
            }
            if req.return_documents:
                item["document"] = (
                    {"text": doc_texts[orig_idx]}
                    if isinstance(req.documents[orig_idx], str)
                    else req.documents[orig_idx]
                )
            results.append(item)

        log.info(
            f"✅ [Rerank] Scored {len(req.documents)} docs → top {len(results)} "
            f"| top_score={results[0]['relevance_score'] if results else 0.0:.4f} "
            f"| tokens={total_tokens} | request_id={request_id}"
        )

        return {
            "id": f"rerank-{uuid.uuid4().hex[:12]}",
            "results": results,
            "model": req.model,
            "usage": {
                "total_tokens": total_tokens,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"rerank failed | request_id={request_id}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "rerank_failed",
                "message": "Document reranking failed, please check server logs",
                "request_id": request_id,
            },
        )
