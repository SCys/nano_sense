import base64
import io
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
from fastapi import APIRouter, Body, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from loguru import logger
from pydantic import BaseModel, Field

from globals import get_tts_model

router = APIRouter()

# OpenAI 兼容音色名 → 适用于 VoxCPM2 的自然语言音色描述
OPENAI_VOICE_PRESETS = {
    "alloy":   "清晰自然、语调平稳中性的声音",
    "ash":     "温暖、略带沙哑、沉稳自信的男声",
    "ballad":  "富有表现力、讲述感强、亲切温和的男声",
    "coral":   "活泼开朗、亲切热情的年轻女声",
    "echo":    "沉着冷静、节奏舒缓平稳的男声",
    "fable":   "富有感染力、生动讲故事风格的声音",
    "onyx":    "低沉浑厚、富有磁性和权威感的男声",
    "nova":    "充满活力、明亮清脆的年轻女声",
    "sage":    "成熟知性、温和沉静的女声",
    "shimmer": "温柔甜美、轻柔细腻的女声",
    "verse":   "灵动多变、富有韵律感和表现力的声音",
}

# response_format → (soundfile 格式名, MIME 类型)
_SF_FORMATS = {
    "wav":  ("WAV", "audio/wav"),
    "mp3":  ("MP3", "audio/mpeg"),
    "flac": ("FLAC", "audio/flac"),
    "ogg":  ("OGG", "audio/ogg"),
}


class SpeechRequest(BaseModel):
    """OpenAI 兼容的语音生成请求体，支持声音设计与克隆扩展"""

    model: str = Field(default="voxcpm2", description="模型名，兼容性字段")
    input: str = Field(..., description="要合成的文本")
    voice: str = Field(
        default="alloy",
        description=(
            "音色：OpenAI 预设音色名（alloy/ash/ballad/coral/echo/fable/"
            "onyx/nova/sage/shimmer/verse），或任意自然语言音色描述"
        ),
    )
    response_format: str = Field(default="mp3", description="输出格式：mp3/wav/flac/ogg/opus/aac/pcm")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="语速")
    instructions: Optional[str] = Field(
        default=None,
        description="额外的语气/风格/情绪控制指令（如：稍微快一点，兴奋开心的语气）",
    )
    reference_audio: Optional[str] = Field(
        default=None,
        description="用于声音克隆的参考音频（支持 base64 编码数据或 data URL）",
    )
    prompt_text: Optional[str] = Field(
        default=None,
        description="参考音频对应的原文本内容（提供后开启高保真极致克隆模式）",
    )
    cfg_value: float = Field(default=2.0, ge=1.0, le=5.0, description="生成引导强度")
    inference_timesteps: int = Field(default=10, ge=4, le=30, description="扩散推理步数")


def get_request_id(request: Request) -> str:
    """从请求头获取或生成请求ID"""
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = str(uuid.uuid4())
    return request_id


def resolve_voice_instruct(voice: Optional[str], instructions: Optional[str]) -> Optional[str]:
    """把 OpenAI 风格的 voice/instructions 解析为 VoxCPM2 格式的音色描述。"""
    parts = []
    if voice:
        preset = OPENAI_VOICE_PRESETS.get(voice.strip().lower())
        if preset:
            parts.append(preset)
        else:
            # 允许直接传自然语言音色描述，如 "温柔甜美的年轻女性声音"
            parts.append(voice.strip())
    if instructions:
        parts.append(instructions.strip())
    return "，".join(parts) if parts else None


def build_voxcpm_text(text: str, voice_instruct: Optional[str]) -> str:
    """构造 VoxCPM2 接收的带音色提示词文本。

    VoxCPM2 规则：将音色描述用圆括号置于文本开头，如 (温柔女声)你好。
    """
    clean_text = text.strip()
    if voice_instruct and voice_instruct.strip():
        return f"({voice_instruct.strip()}){clean_text}"
    return clean_text


def encode_audio(wav: np.ndarray, sample_rate: int, response_format: str) -> Tuple[bytes, str]:
    """把 float32 波形编码为目标格式。

    支持: wav / mp3 / flac / ogg（libsndfile 原生）、opus / aac（ffmpeg 兜底）、
    pcm（16-bit 小端单声道原始样本，同 OpenAI 定义）。
    """
    fmt = response_format.lower().strip(".")
    if fmt == "oga":
        fmt = "ogg"

    if fmt == "pcm":
        pcm = (np.clip(wav, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        return pcm, "audio/pcm"

    if fmt in _SF_FORMATS:
        sf_fmt, media_type = _SF_FORMATS[fmt]
        buffer = io.BytesIO()
        sf.write(buffer, wav, sample_rate, format=sf_fmt)
        buffer.seek(0)
        return buffer.getvalue(), media_type

    if fmt in ("opus", "aac"):
        return _encode_with_ffmpeg(wav, sample_rate, fmt)

    raise ValueError(f"Unsupported response_format '{response_format}'")


def _encode_with_ffmpeg(wav: np.ndarray, sample_rate: int, fmt: str) -> Tuple[bytes, str]:
    """用系统 ffmpeg 编码 opus / aac"""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(f"fmt '{fmt}' requires ffmpeg on PATH")

    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0"
    ]
    if fmt == "opus":
        cmd += ["-c:a", "libopus", "-b:a", "64k", "-f", "ogg", "pipe:1"]
        media_type = "audio/ogg"
    else:  # aac (ADTS 流)
        cmd += ["-c:a", "aac", "-b:a", "128k", "-f", "adts", "pipe:1"]
        media_type = "audio/aac"

    result = subprocess.run(cmd, input=wav.tobytes(), capture_output=True)
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(f"ffmpeg encoding failed: {result.stderr.decode(errors='ignore')[:500]}")
    return result.stdout, media_type


def parse_audio_data(audio_data: str) -> bytes:
    """解析 base64 字符串或 data URL 为音频字节"""
    if audio_data.startswith("data:"):
        _, data = audio_data.split(",", 1)
    else:
        data = audio_data
    return base64.b64decode(data)


def run_voxcpm_generation(
    model,
    text: str,
    voice_instruct: Optional[str] = None,
    ref_audio_bytes: Optional[bytes] = None,
    prompt_text: Optional[str] = None,
    speed: float = 1.0,
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
) -> Tuple[np.ndarray, int]:
    """调用 VoxCPM2 执行语音生成（支持声音设计、克隆与复合控制）"""
    sample_rate = getattr(model.tts_model, "sample_rate", 48000)
    styled_text = build_voxcpm_text(text, voice_instruct)

    temp_ref_path = None
    try:
        if ref_audio_bytes:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(ref_audio_bytes)
                temp_ref_path = tmp.name

        if temp_ref_path and prompt_text:
            # 极致克隆模式 (Ultimate Cloning)
            wav = model.generate(
                text=styled_text,
                prompt_wav_path=temp_ref_path,
                prompt_text=prompt_text,
                reference_wav_path=temp_ref_path,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
            )
        elif temp_ref_path:
            # 标准声音克隆 / 复合控制克隆 (Voice Cloning / Controllable Cloning)
            wav = model.generate(
                text=styled_text,
                reference_wav_path=temp_ref_path,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
            )
        else:
            # 零样本 / 声音设计模式 (Zero-shot / Voice Design)
            wav = model.generate(
                text=styled_text,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
            )

        # 语速调节（相位声码器时间拉伸，不变调）
        if speed != 1.0:
            wav = librosa.effects.time_stretch(y=wav.astype(np.float32), rate=speed)

        return wav.astype(np.float32), sample_rate

    finally:
        if temp_ref_path and os.path.exists(temp_ref_path):
            try:
                os.unlink(temp_ref_path)
            except Exception:
                pass


def _synthesize_response(
    request: Request,
    log,
    text: str,
    voice_instruct: Optional[str],
    ref_audio_bytes: Optional[bytes],
    prompt_text: Optional[str],
    speed: float,
    response_format: str,
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
) -> Response:
    """公共合成处理流程"""
    request_id = get_request_id(request)
    start_total = time.perf_counter()

    model = get_tts_model()  # 懒加载 / 超时自动下线保护

    gen_start = time.perf_counter()
    wav, sample_rate = run_voxcpm_generation(
        model=model,
        text=text,
        voice_instruct=voice_instruct,
        ref_audio_bytes=ref_audio_bytes,
        prompt_text=prompt_text,
        speed=speed,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
    )
    gen_ms = (time.perf_counter() - gen_start) * 1000

    audio_bytes, media_type = encode_audio(wav, sample_rate, response_format)
    duration = len(wav) / sample_rate
    total_ms = (time.perf_counter() - start_total) * 1000

    mode = "Clone" if ref_audio_bytes else "Design"
    log.info(
        f"✅ [VoxCPM2-{mode}] Synthesized {len(text)} chars → {len(audio_bytes)} bytes "
        f"| fmt={response_format} | sr={sample_rate}Hz | dur={duration:.2f}s "
        f"| gen={gen_ms:.1f}ms | total={total_ms:.1f}ms "
        f"| voice={voice_instruct or 'default'} | speed={speed} "
        f"| request_id={request_id}"
    )

    filename = f"speech_{uuid.uuid4().hex[:8]}.{response_format.lower().strip('.')}"
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


@router.post("/speech")
async def create_speech(request: Request, req: SpeechRequest = Body(...)):
    """
    OpenAI 兼容的文字转语音与声音克隆接口（POST /v1/audio/speech）

    - **voice**: OpenAI 预设音色名（alloy/nova/shimmer/...），或任意自然语言音色描述
    - **instructions**: 情绪/语气/节奏指令（如“带微笑感，语速稍微放缓”）
    - **reference_audio**: 可选，传入 Base64 编码的参考音频实现一键声音克隆
    - **prompt_text**: 可选，参考音频对应的文本内容，开启极致保真克隆
    - **response_format**: mp3 / wav / flac / ogg / opus / aac / pcm
    - **speed**: 0.25 ~ 4.0 语速调节
    """
    request_id = get_request_id(request)
    log = logger.bind(request_id=request_id)

    if not req.input.strip():
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    fmt = req.response_format.lower().strip(".")
    if fmt == "oga":
        fmt = "ogg"
    supported = set(_SF_FORMATS) | {"opus", "aac", "pcm"}
    if fmt not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format '{req.response_format}'. Supported: {sorted(supported)}",
        )

    ref_bytes = None
    if req.reference_audio:
        try:
            ref_bytes = parse_audio_data(req.reference_audio)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 reference_audio: {e}")

    try:
        voice_instruct = resolve_voice_instruct(req.voice, req.instructions)
        return _synthesize_response(
            request=request,
            log=log,
            text=req.input,
            voice_instruct=voice_instruct,
            ref_audio_bytes=ref_bytes,
            prompt_text=req.prompt_text,
            speed=req.speed,
            response_format=fmt,
            cfg_value=req.cfg_value,
            inference_timesteps=req.inference_timesteps,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception("tts speech failed")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "synthesis_failed",
                "message": str(e),
                "request_id": request_id,
            },
        )


@router.post("/clone")
async def clone_voice_multipart(
    request: Request,
    file: UploadFile = File(..., description="参考音频文件（3~10秒为佳，支持 wav/mp3/ogg/flac/m4a 等）"),
    text: str = Form(..., description="要合成的目标文本"),
    voice: Optional[str] = Form(None, description="可选音色风格补充"),
    instructions: Optional[str] = Form(None, description="可选情绪与语气控制（如：兴奋、悲伤、严肃）"),
    prompt_text: Optional[str] = Form(None, description="参考音频的原文本台词（可选，提供可提升保真度）"),
    response_format: str = Form("mp3", description="输出音频格式"),
    speed: float = Form(1.0, ge=0.25, le=4.0, description="语速"),
    cfg_value: float = Form(2.0, ge=1.0, le=5.0, description="生成引导强度"),
    inference_timesteps: int = Form(10, ge=4, le=30, description="扩散推理步数"),
):
    """
    Multipart 表单声音克隆接口（支持直接上传音频文件）

    上传说话人的一小段音频文件，直接克隆该声音朗读新文本，并支持文字控制语气情绪。
    """
    request_id = get_request_id(request)
    log = logger.bind(request_id=request_id)

    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    fmt = response_format.lower().strip(".")
    if fmt == "oga":
        fmt = "ogg"
    supported = set(_SF_FORMATS) | {"opus", "aac", "pcm"}
    if fmt not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format '{response_format}'. Supported: {sorted(supported)}",
        )

    try:
        ref_bytes = await file.read()
        if not ref_bytes:
            raise HTTPException(status_code=400, detail="Uploaded reference file is empty")

        voice_instruct = resolve_voice_instruct(voice, instructions)
        return _synthesize_response(
            request=request,
            log=log,
            text=text,
            voice_instruct=voice_instruct,
            ref_audio_bytes=ref_bytes,
            prompt_text=prompt_text,
            speed=speed,
            response_format=fmt,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception("tts clone failed")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "synthesis_failed",
                "message": str(e),
                "request_id": request_id,
            },
        )


@router.post("/synthesize")
async def synthesize_speech(
    request: Request,
    text: str = Query(..., description="要合成的文本"),
    voice: Optional[str] = Query(None, description="音色描述，如：温柔甜美的女性声音"),
    speed: float = Query(1.0, ge=0.25, le=4.0, description="语速"),
    response_format: str = Query("wav", description="输出格式，当前支持 wav/mp3/flac/ogg/pcm"),
):
    """
    Query 参数文字转语音接口（兼容旧版）
    """
    request_id = get_request_id(request)
    log = logger.bind(request_id=request_id)

    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    fmt = response_format.lower().strip(".")
    if fmt == "oga":
        fmt = "ogg"
    supported = set(_SF_FORMATS) | {"opus", "aac", "pcm"}
    if fmt not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format '{response_format}'. Supported: {sorted(supported)}",
        )

    try:
        return _synthesize_response(
            request=request,
            log=log,
            text=text,
            voice_instruct=resolve_voice_instruct(voice, None),
            ref_audio_bytes=None,
            prompt_text=None,
            speed=speed,
            response_format=fmt,
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
