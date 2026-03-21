from fastapi import FastAPI, UploadFile, File
import sherpa_onnx
import uvicorn
import tempfile
import os
import librosa
import numpy as np
import time
from pathlib import Path

app = FastAPI(title="FireRedASR2-CTC OpenAI API（Docker 隔离版）")

# ================== 容器内路径（固定） ==================
MODEL_DIR = "/app/models/sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25"
PORT = 3025

recognizer = sherpa_onnx.OfflineRecognizer.from_fire_red_asr_ctc(
    model=f"{MODEL_DIR}/model.int8.onnx",
    tokens=f"{MODEL_DIR}/tokens.txt",
    num_threads=4,                    # Docker 内建议保持 4（可通过 compose 调整 CPU）
)

print(f"✅ FireRedASR2-CTC int8 加载成功 (v{sherpa_onnx.__version__}) - 端口 {PORT}")

def load_audio(filename: str) -> np.ndarray:
    audio, sr = librosa.load(filename, sr=16000, mono=True)
    assert sr == 16000
    return np.ascontiguousarray(audio, dtype=np.float32)

def human_readable_size(num_bytes: int) -> str:
    if num_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(np.log(num_bytes) / np.log(1024))
    p = 1024 ** i
    s = round(num_bytes / p, 2)
    return f"{s} {size_name[i]}"

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    response_format: str = "json"
):
    start_total = time.perf_counter()
    content = await file.read()
    file_size = len(content)
    file_type = file.content_type or f"audio/{Path(file.filename).suffix.lower().lstrip('.') or 'unknown'}"
    size_human = human_readable_size(file_size)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    receive_ms = (time.perf_counter() - start_total) * 1000

    try:
        load_start = time.perf_counter()
        samples = load_audio(tmp_path)
        load_ms = (time.perf_counter() - load_start) * 1000

        decode_start = time.perf_counter()
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate=16000, waveform=samples)
        recognizer.decode_stream(stream)
        decode_ms = (time.perf_counter() - decode_start) * 1000

        text = stream.result.text
        duration = len(samples) / 16000
        total_ms = (time.perf_counter() - start_total) * 1000
        rtf = decode_ms / 1000 / duration if duration > 0 else 0.0

        # 单行日志（干净）
        print(
            f"✅ [FireRedASR2-CTC] Transcribed {duration:.2f}s "
            f"| size={size_human} "
            f"| receive={receive_ms:.1f}ms "
            f"| load={load_ms:.1f}ms "
            f"| decode={decode_ms:.1f}ms "
            f"| total={total_ms:.1f}ms "
            f"| RTF={rtf:.3f} "
            f"| type={file_type}"
        )

        result = {
            "text": text,
            "duration_seconds": round(duration, 4),
            "file_size_bytes": file_size,
            "file_size_human": size_human,
            "file_type": file_type,
            "timings": {
                "receive_ms": round(receive_ms, 2),
                "load_audio_ms": round(load_ms, 2),
                "decode_ms": round(decode_ms, 2),
                "total_ms": round(total_ms, 2),
                "rtf": round(rtf, 3)
            }
        }
        if response_format == "verbose_json":
            result["segments"] = [{"start": 0, "end": round(duration, 4), "text": text}]

        return result

    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)