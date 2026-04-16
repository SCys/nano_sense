import sys
import unittest
from unittest.mock import MagicMock
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ============ Mock funasr before imports ============
mock_recognizer = MagicMock()
mock_recognizer.generate.return_value = [{"text": "这是测试文本"}]

mock_funasr = MagicMock()
mock_funasr.AutoModel.return_value = mock_recognizer
mock_funasr.__version__ = "1.0.0"
sys.modules["funasr"] = mock_funasr

# ============ Mock librosa ============
mock_librosa = MagicMock()
# Return 1 second of audio (16000 samples at 16000 Hz)
mock_librosa.load.return_value = (np.array([0.0] * 16000, dtype=np.float32), 16000)
sys.modules["librosa"] = mock_librosa

import api_audio_transcriptions

# 避免测试依赖 globals 中的真实模型加载逻辑
api_audio_transcriptions.get_asr_recognizer = MagicMock(return_value=mock_recognizer)
audio_router = api_audio_transcriptions.router

# Build minimal FastAPI app with only audio router
app = FastAPI()
app.include_router(audio_router, prefix="/v1/audio")


class TestAPIAudioTranscriptionsMock(unittest.TestCase):
    """使用模拟测试音频转录API（FastAPI版本）"""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_transcription_json_format(self):
        """测试JSON格式的音频转录"""
        response = self.client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.ogg", b"fake audio", "audio/ogg")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("text", data)
        self.assertEqual(data["text"], "这是测试文本")
        # Verify recognizer was used
        mock_recognizer.generate.assert_called()

    def test_transcription_text_format(self):
        """测试TEXT格式的音频转录"""
        response = self.client.post(
            "/v1/audio/transcriptions?response_format=text",
            files={"file": ("test.ogg", b"fake audio", "audio/ogg")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), "这是测试文本")

    def test_transcription_with_segments(self):
        """测试带分段信息的音频转录（新实现只返回单段）"""
        response = self.client.post(
            "/v1/audio/transcriptions?timestamp_granularities=segment",
            files={"file": ("test.ogg", b"fake audio", "audio/ogg")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("text", data)
        self.assertIn("segments", data)
        self.assertEqual(len(data["segments"]), 1)
        seg = data["segments"][0]
        self.assertEqual(seg["start"], 0.0)
        self.assertEqual(seg["end"], 1.0)  # 1 second duration from mocked audio
        self.assertEqual(seg["text"], "这是测试文本")

    def test_exception_handling(self):
        """测试异常处理"""
        # Make recognizer raise exception
        mock_recognizer.generate.side_effect = Exception("测试异常")
        response = self.client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.ogg", b"fake audio", "audio/ogg")},
        )
        self.assertEqual(response.status_code, 500)
        data = response.json()
        # FastAPI wraps HTTPException details in "detail" key
        self.assertIn("detail", data)
        detail = data["detail"]
        self.assertEqual(detail["error"], "transcription_failed")
        self.assertIn("request_id", detail)
        # Reset side effect for other tests
        mock_recognizer.generate.side_effect = None


if __name__ == "__main__":
    unittest.main()
