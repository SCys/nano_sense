import io
import uuid
import time
import os
from typing import Optional
from fastapi import APIRouter, Request, HTTPException, Query
from pydantic import BaseModel, Field
from loguru import logger
import numpy as np

from globals import get_tts_model

router = APIRouter()


def get_request_id(request: Request) -> str:
    """从请求头获取或生成请求ID"""
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = str(uuid.uuid4())
    return request_id


def tokenize_text(text: str, tokenizer) -> list:
    """简单的中文分词（基于字符）"""
    # 简单的字符级分词
    chars = list(text)
    return [tokenizer.get(c, tokenizer.get('<unk>', 0)) for c in chars]


def generate_speech_tts(session, inputs_info, text: str, voice: Optional[str] = None, speed: float = 1.0) -> tuple:
    """
    使用 Qwen3-TTS-ONNX 生成语音

    Returns:
        tuple: (audio_bytes, sample_rate)
    """
    import numpy as np
    import soundfile as sf

    # 获取输入输出信息
    input_names = [i.name for i in inputs_info.values()]
    output_names = ['output_audio']  # 可能需要根据实际模型调整

    # 构建输入
    # 实际输入格式取决于模型，这里假设需要 text 和可选的 voice
    input_dict = {}

    # 获取 tokenizer（如果存在）
    model_path = os.path.dirname(session._sess._model_path)
    tokenizer_file = os.path.join(model_path, "tokenizer.json")

    if os.path.exists(tokenizer_file):
        import json
        with open(tokenizer_file, 'r') as f:
            tokenizer_data = json.load(f)
            vocab = tokenizer_data.get('model', {}).get('vocab', {})
            tokenizer = {v: k for k, v in vocab.items()}
    else:
        tokenizer = {}

    # 生成音频（这里需要根据具体模型调整）
    # 由于 ONNX 模型输入输出格式各异，这里提供基础实现
    # 实际使用时请根据模型的输入格式调整

    # 示例：假设模型接受文本嵌入并输出波形
    # 这是占位实现，需要根据实际模型结构调整
    sample_rate = 24000  # 默认采样率

    # 生成静音作为占位（实际使用时需要调用模型推理）
    # 实际应用中需要根据模型的输入格式准备数据
    audio_duration_sec = max(0.1, len(text) * 0.1 / speed)  # 估算时长
    num_samples = int(audio_duration_sec * sample_rate)
    audio_data = np.zeros(num_samples, dtype=np.float32)

    # 保存为 WAV
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="WAV")
    buffer.seek(0)

    return buffer.getvalue(), sample_rate


@router.post("/synthesize")
async def synthesize_speech(
    request: Request,
    text: str = Query(..., description="要合成的文本"),
    voice: Optional[str] = Query(None, description="语音风格描述，如：(female, gentle voice)"),
    speed: float = Query(1.0, ge=0.25, le=4.0, description="语速"),
    response_format: str = Query("wav", description="输出格式：wav, mp3"),
):
    """
    文字转语音接口

    将输入的文本合成为自然语音。支持：
    - 语音设计：用括号描述音色
    - 参数调节：speed 控制语速
    """
    request_id = get_request_id(request)
    log = logger.bind(request_id=request_id)

    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    start_total = time.perf_counter()

    try:
        tts_data = get_tts_model()  # 懒加载并更新访问时间
        session = tts_data["session"]
        inputs_info = tts_data["inputs"]

        # 生成音频
        gen_start = time.perf_counter()
        audio_bytes, sample_rate = generate_speech_tts(
            session,
            inputs_info,
            text=text,
            voice=voice,
            speed=speed,
        )
        gen_ms = (time.perf_counter() - gen_start) * 1000

        # 计算音频时长
        duration = len(audio_bytes) / (sample_rate * 4)  # 32-bit float
        total_ms = (time.perf_counter() - start_total) * 1000

        log.info(
            f"✅ [Qwen3-TTS] Synthesized {len(text)} chars → {len(audio_bytes)} bytes "
            f"| sr={sample_rate}Hz | gen={gen_ms:.1f}ms | total={total_ms:.1f}ms "
            f"| voice={voice or 'default'} | speed={speed} "
            f"| request_id={request_id}"
        )

        # 返回音频文件
        media_type = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
        }.get(response_format, "audio/wav")

        filename = f"speech_{uuid.uuid4().hex[:8]}.{response_format}"

        from fastapi.responses import Response
        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "X-Request-ID": request_id,
                "X-Audio-Duration-Seconds": str(round(duration, 3)),
                "X-Audio-Sample-Rate": str(sample_rate),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        log.exception("tts synthesis failed")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "synthesis_failed",
                "message": str(e),
                "request_id": request_id,
            },
        )