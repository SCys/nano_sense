import io
import time
import unittest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
import wave

from api_audio_synthesis import router as synthesis_router


class MockVocoder:
    """模拟 Vocoder"""
    def decode(self, tokens):
        # 返回模拟的音频波形
        import torch
        return torch.randn(1, 24000)  # 1秒 24kHz


class TestAudioSynthesis(unittest.TestCase):
    """测试文字转音频API"""

    @classmethod
    def setUpClass(cls):
        app = FastAPI()
        app.include_router(synthesis_router, prefix="/v1/audio")
        cls.client = TestClient(app)

    @patch("api_audio_synthesis.get_tts_model")
    def test_synthesis_success(self, mock_get_tts):
        """测试成功的语音合成"""
        # 模拟 TTS 模型
        mock_tts_data = {
            "model": MagicMock(),
            "tokenizer": MagicMock(),
            "vocoder": MockVocoder(),
        }
        mock_get_tts.return_value = mock_tts_data

        # 模拟 tokenizer 和生成
        with patch("api_audio_synthesis.prepare_input_ids") as mock_prepare:
            mock_prepare.return_value = [1, 2, 3, 4]
            with patch("api_audio_synthesis.generate_speech") as mock_generate:
                # 生成一个简单的 WAV 音频（1秒静音）
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(24000)
                    wf.writeframes(b'\x00' * 48000)  # 1秒静音
                mock_generate.return_value = buffer.getvalue()

                response = self.client.get(
                    "/v1/audio/synthesize",
                    params={"text": "你好，这是一个测试。"}
                )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["content-type"], "audio/wav")
                self.assertIn("X-Request-ID", response.headers)
                self.assertIn("X-Audio-Duration-Seconds", response.headers)

    def test_empty_text(self):
        """测试空文本"""
        response = self.client.get(
            "/v1/audio/synthesize",
            params={"text": "   "}
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["detail"], "Text cannot be empty")

    @patch("api_audio_synthesis.get_tts_model")
    def test_synthesis_model_error(self, mock_get_tts):
        """测试模型内部错误"""
        mock_get_tts.side_effect = RuntimeError("CUDA out of memory")

        response = self.client.get(
            "/v1/audio/synthesize",
            params={"text": "测试"}
        )

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn("detail", data)
        self.assertEqual(data["detail"]["error"], "synthesis_failed")
        self.assertIn("request_id", data["detail"])

    @patch("api_audio_synthesis.get_tts_model")
    def test_synthesis_different_formats(self, mock_get_tts):
        """测试不同输出格式"""
        mock_tts_data = {
            "model": MagicMock(),
            "tokenizer": MagicMock(),
            "vocoder": MockVocoder(),
        }
        mock_get_tts.return_value = mock_tts_data

        with patch("api_audio_synthesis.prepare_input_ids"):
            with patch("api_audio_synthesis.generate_speech") as mock_generate:
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(24000)
                    wf.writeframes(b'\x00' * 48000)
                mock_generate.return_value = buffer.getvalue()

                for fmt in ["mp3", "wav", "opus", "flac"]:
                    response = self.client.get(
                        "/v1/audio/synthesize",
                        params={"text": "测试", "response_format": fmt}
                    )
                    self.assertEqual(response.status_code, 200)
                    expected_type = {
                        "mp3": "audio/mpeg",
                        "wav": "audio/wav",
                        "opus": "audio/opus",
                        "flac": "audio/flac",
                    }[fmt]
                    self.assertEqual(response.headers["content-type"], expected_type)

    @patch("api_audio_synthesis.get_tts_model")
    def test_synthesis_with_params(self, mock_get_tts):
        """测试带参数的合成请求"""
        mock_tts_data = {
            "model": MagicMock(),
            "tokenizer": MagicMock(),
            "vocoder": MockVocoder(),
        }
        mock_get_tts.return_value = mock_tts_data

        with patch("api_audio_synthesis.prepare_input_ids"):
            with patch("api_audio_synthesis.generate_speech") as mock_generate:
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(24000)
                    wf.writeframes(b'\x00' * 48000)
                mock_generate.return_value = buffer.getvalue()

                response = self.client.get(
                    "/v1/audio/synthesize",
                    params={
                        "text": "测试",
                        "voice": "female",
                        "speed": 1.5,
                        "response_format": "wav"
                    }
                )

                self.assertEqual(response.status_code, 200)
                # 验证参数传递
                call_kwargs = mock_generate.call_args[1]
                self.assertEqual(call_kwargs["voice"], "female")
                self.assertEqual(call_kwargs["speed"], 1.5)


if __name__ == "__main__":
    unittest.main()
