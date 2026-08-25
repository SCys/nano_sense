import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse
from loguru import logger
from pydantic import BaseModel, Field

from globals import use_asr_recognizer

router = APIRouter()

MAX_AUDIO_BYTES = 100 * 1024 * 1024  # 100MB


class TranscriptionRequest(BaseModel):
    """音频转录请求模型"""
    response_format: str = Field("json", description="响应格式: json, text, verbose_json")
    timestamp_granularities: Optional[str] = Field(None, description="时间戳粒度")


def get_request_id(request: Request) -> str:
    """从请求头获取或生成请求ID"""
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = str(uuid.uuid4())
    return request_id


def extract_text_from_funasr_result(asr_result) -> str:
    """从 FunASR 返回结果中提取文本。"""
    if isinstance(asr_result, str):
        return asr_result.strip()

    if isinstance(asr_result, dict):
        text = asr_result.get("text", "")
        return str(text).strip()

    if isinstance(asr_result, list):
        chunks: list[str] = []
        for item in asr_result:
            if isinstance(item, dict):
                chunk_text = item.get("text", "")
                if chunk_text:
                    chunks.append(str(chunk_text).strip())
            elif isinstance(item, str) and item.strip():
                chunks.append(item.strip())
        return "".join([chunk for chunk in chunks if chunk]).strip()

    return str(asr_result).strip()


def load_audio(filename: str) -> np.ndarray:
    audio, sr = librosa.load(filename, sr=16000, mono=True)
    if sr != 16000:
        raise RuntimeError(f"librosa failed to resample audio to 16kHz (got {sr}Hz)")
    return np.ascontiguousarray(audio, dtype=np.float32)


def human_readable_size(num_bytes: int) -> str:
    if num_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(np.log(num_bytes) / np.log(1024))
    p = 1024**i
    s = round(num_bytes / p, 2)
    return f"{s} {size_name[i]}"


def _sync_asr_pipeline(recognizer, tmp_path: str) -> Tuple[str, float, float, float]:
    """在工作线程池中执行音频读取与声学模型解码"""
    load_start = time.perf_counter()
    samples = load_audio(tmp_path)
    load_ms = (time.perf_counter() - load_start) * 1000

    decode_start = time.perf_counter()
    asr_result = recognizer.generate(input=tmp_path)
    decode_ms = (time.perf_counter() - decode_start) * 1000

    text = extract_text_from_funasr_result(asr_result)
    duration = len(samples) / 16000
    return text, duration, load_ms, decode_ms


@router.post("/transcriptions")
async def transcribe(
    request: Request,
    file: UploadFile | None = File(None),
    query_params: TranscriptionRequest = Depends(),
):
    """
    音频转录接口

    - **response_format**: 响应格式 (json, text, verbose_json)
    - **timestamp_granularities**: 时间戳粒度（如 segment）
    """
    request_id = get_request_id(request)
    log = logger.bind(request_id=request_id)

    # 兼容无文件请求
    if file is None:
        return {"text": ""}

    response_format = query_params.response_format
    timestamp_granularities = query_params.timestamp_granularities

    start_total = time.perf_counter()
    content = await file.read()
    file_size = len(content)

    if file_size > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Audio file exceeds maximum allowed size of {MAX_AUDIO_BYTES // 1048576}MB",
        )

    file_type = file.content_type or f"audio/{Path(file.filename or '').suffix.lower().lstrip('.') or 'unknown'}"
    size_human = human_readable_size(file_size)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        receive_ms = (time.perf_counter() - start_total) * 1000

        # 通过 use_asr_recognizer 保护，并卸载至工作线程池防止阻塞主事件循环
        with use_asr_recognizer() as recognizer:
            text, duration, load_ms, decode_ms = await run_in_threadpool(
                _sync_asr_pipeline,
                recognizer=recognizer,
                tmp_path=tmp_path,
            )

        total_ms = (time.perf_counter() - start_total) * 1000
        rtf = decode_ms / 1000 / duration if duration > 0 else 0.0

        log.info(
            f"✅ [FunASR] Transcribed {duration:.2f}s "
            f"| size={size_human} "
            f"| receive={receive_ms:.1f}ms "
            f"| load={load_ms:.1f}ms "
            f"| decode={decode_ms:.1f}ms "
            f"| total={total_ms:.1f}ms "
            f"| RTF={rtf:.3f} "
            f"| type={file_type} "
            f"| request_id={request_id}"
        )

        need_segments = (response_format == "verbose_json") or (timestamp_granularities is not None)

        if response_format == "text":
            return PlainTextResponse(content=text)

        result: dict = {"text": text}
        if need_segments:
            result["segments"] = [{"id": 0, "start": 0.0, "end": round(duration, 4), "text": text}]
            if response_format == "verbose_json":
                result.update(
                    {
                        "duration_seconds": round(duration, 4),
                        "file_size_bytes": file_size,
                        "file_size_human": size_human,
                        "file_type": file_type,
                        "timings": {
                            "receive_ms": round(receive_ms, 2),
                            "load_audio_ms": round(load_ms, 2),
                            "decode_ms": round(decode_ms, 2),
                            "total_ms": round(total_ms, 2),
                            "rtf": round(rtf, 3),
                        },
                    }
                )
        return result

    except HTTPException:
        raise
    except Exception as e:
        log.exception("asr failed")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "transcription_failed",
                "message": "ASR transcription failed, please check server logs",
                "request_id": request_id,
            },
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as e:
                log.warning(f"Failed to delete temp file {tmp_path}: {e}")
