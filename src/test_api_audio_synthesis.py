import base64
import io
import unittest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

import numpy as np
import soundfile as sf

from api_audio_synthesis import (
    OPENAI_VOICE_PRESETS,
    build_voxcpm_text,
    resolve_voice_instruct,
    router as synthesis_router,
)


def make_wave(duration_sec: float = 1.0, sample_rate: int = 48000) -> np.ndarray:
    """生成正弦波 float32 波形 (48kHz)"""
    t = np.linspace(0, duration_sec, int(duration_sec * sample_rate), endpoint=False)
    return (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def make_mock_voxcpm_model():
    """模拟 VoxCPM 模型实例"""
    mock = MagicMock()
    mock.tts_model.sample_rate = 48000
    mock.generate.return_value = make_wave(1.0, 48000)
    return mock


class TestResolveVoiceInstruct(unittest.TestCase):
    """测试音色参数解析与 VoxCPM 格式构造"""

    def test_preset_voice(self):
        self.assertEqual(
            resolve_voice_instruct("nova", None),
            OPENAI_VOICE_PRESETS["nova"],
        )

    def test_preset_voice_with_instructions(self):
        result = resolve_voice_instruct("alloy", "语速放慢")
        self.assertTrue(result.startswith(OPENAI_VOICE_PRESETS["alloy"]))
        self.assertIn("语速放慢", result)

    def test_custom_voice_as_description(self):
        self.assertEqual(
            resolve_voice_instruct("温柔甜美的女声", None),
            "温柔甜美的女声",
        )

    def test_build_voxcpm_text(self):
        self.assertEqual(
            build_voxcpm_text("你好", "温柔女声"),
            "(温柔女声)你好",
        )
        self.assertEqual(
            build_voxcpm_text("你好", None),
            "你好",
        )


class TestOpenAISpeech(unittest.TestCase):
    """测试 OpenAI 兼容接口（POST /v1/audio/speech）"""

    @classmethod
    def setUpClass(cls):
        app = FastAPI()
        app.include_router(synthesis_router, prefix="/v1/audio")
        cls.client = TestClient(app)

    @patch("api_audio_synthesis.get_tts_model")
    def test_speech_mp3_default(self, mock_get_tts):
        """默认返回 48kHz mp3"""
        mock_model = make_mock_voxcpm_model()
        mock_get_tts.return_value = mock_model

        resp = self.client.post(
            "/v1/audio/speech",
            json={"model": "voxcpm2", "input": "你好世界", "voice": "nova"},
        )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "audio/mpeg")
        self.assertEqual(resp.headers["X-Audio-Sample-Rate"], "48000")

        # 真实编码出的 mp3 可正常读回
        data, sr = sf.read(io.BytesIO(resp.content))
        self.assertEqual(sr, 48000)
        self.assertAlmostEqual(len(data) / sr, 1.0, places=1)

        # 验证调用文本带括号提示词
        args, kwargs = mock_model.generate.call_args
        self.assertIn(OPENAI_VOICE_PRESETS["nova"], kwargs["text"])

    @patch("api_audio_synthesis.get_tts_model")
    def test_speech_wav_format(self, mock_get_tts):
        """wav 格式输出"""
        mock_get_tts.return_value = make_mock_voxcpm_model()

        resp = self.client.post(
            "/v1/audio/speech",
            json={"input": "hello", "voice": "alloy", "response_format": "wav"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "audio/wav")

    @patch("api_audio_synthesis.get_tts_model")
    def test_speech_pcm_format(self, mock_get_tts):
        """pcm 16-bit 原始采样字节"""
        mock_get_tts.return_value = make_mock_voxcpm_model()

        resp = self.client.post(
            "/v1/audio/speech",
            json={"input": "hello", "voice": "onyx", "response_format": "pcm"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "audio/pcm")
        # 1 秒 48000 个采样 * 2 字节 = 96000 字节
        self.assertEqual(len(resp.content), 48000 * 2)

    @patch("api_audio_synthesis.get_tts_model")
    def test_speech_base64_clone(self, mock_get_tts):
        """通过 base64 传入参考音频执行声音克隆"""
        mock_model = make_mock_voxcpm_model()
        mock_get_tts.return_value = mock_model

        # 生成 0.5s 测试音频作为 base64 参考
        buf = io.BytesIO()
        sf.write(buf, make_wave(0.5, 48000), 48000, format="WAV")
        b64_audio = base64.b64encode(buf.getvalue()).decode("utf-8")

        resp = self.client.post(
            "/v1/audio/speech",
            json={
                "input": "克隆测试",
                "reference_audio": f"data:audio/wav;base64,{b64_audio}",
                "instructions": "开心的语气",
            },
        )
        self.assertEqual(resp.status_code, 200)
        args, kwargs = mock_model.generate.call_args
        self.assertIn("reference_wav_path", kwargs)
        self.assertTrue(kwargs["reference_wav_path"] is not None)
        self.assertIn("开心的语气", kwargs["text"])

    def test_speech_empty_input(self):
        resp = self.client.post("/v1/audio/speech", json={"input": "   "})
        self.assertEqual(resp.status_code, 400)

    def test_speech_unsupported_format(self):
        resp = self.client.post(
            "/v1/audio/speech",
            json={"input": "hello", "response_format": "wma"},
        )
        self.assertEqual(resp.status_code, 400)

    @patch("api_audio_synthesis.get_tts_model")
    def test_speech_model_error(self, mock_get_tts):
        mock_get_tts.side_effect = RuntimeError("CUDA OOM")

        resp = self.client.post(
            "/v1/audio/speech",
            json={"input": "hello"},
        )
        self.assertEqual(resp.status_code, 500)
        data = resp.json()
        self.assertEqual(data["detail"]["error"], "synthesis_failed")


class TestVoiceCloneMultipart(unittest.TestCase):
    """测试 Multipart 上传文件的声音克隆接口（POST /v1/audio/clone）"""

    @classmethod
    def setUpClass(cls):
        app = FastAPI()
        app.include_router(synthesis_router, prefix="/v1/audio")
        cls.client = TestClient(app)

    @patch("api_audio_synthesis.get_tts_model")
    def test_clone_multipart_success(self, mock_get_tts):
        mock_model = make_mock_voxcpm_model()
        mock_get_tts.return_value = mock_model

        buf = io.BytesIO()
        sf.write(buf, make_wave(0.5, 48000), 48000, format="WAV")
        raw_bytes = buf.getvalue()

        resp = self.client.post(
            "/v1/audio/clone",
            data={
                "text": "直接上传文件的克隆测试",
                "instructions": "更深沉一些",
                "response_format": "wav",
            },
            files={"file": ("speaker.wav", raw_bytes, "audio/wav")},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "audio/wav")

        args, kwargs = mock_model.generate.call_args
        self.assertIn("reference_wav_path", kwargs)
        self.assertIn("更深沉一些", kwargs["text"])

    def test_clone_empty_text(self):
        resp = self.client.post(
            "/v1/audio/clone",
            data={"text": "   "},
            files={"file": ("speaker.wav", b"fake", "audio/wav")},
        )
        self.assertEqual(resp.status_code, 400)


class TestSynthesizeBackwardCompat(unittest.TestCase):
    """测试兼容旧版的 Query 参数接口（POST /v1/audio/synthesize）"""

    @classmethod
    def setUpClass(cls):
        app = FastAPI()
        app.include_router(synthesis_router, prefix="/v1/audio")
        cls.client = TestClient(app)

    @patch("api_audio_synthesis.get_tts_model")
    def test_synthesize_query_params(self, mock_get_tts):
        mock_model = make_mock_voxcpm_model()
        mock_get_tts.return_value = mock_model

        resp = self.client.post(
            "/v1/audio/synthesize",
            params={"text": "兼容测试", "voice": "温柔女声", "response_format": "wav"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "audio/wav")


if __name__ == "__main__":
    unittest.main()
