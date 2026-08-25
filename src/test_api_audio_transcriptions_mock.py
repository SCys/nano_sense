import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api_audio_transcriptions import router as audio_router


class TestAPIAudioTranscriptionsMock(unittest.TestCase):
    """使用模拟测试音频转录API"""

    @classmethod
    def setUpClass(cls):
        app = FastAPI()
        app.include_router(audio_router, prefix="/v1/audio")
        cls.client = TestClient(app)

    @patch("api_audio_transcriptions.load_audio")
    @patch("api_audio_transcriptions.get_asr_recognizer")
    def test_transcription_json_format(self, mock_get_asr, mock_load):
        mock_rec = MagicMock()
        mock_rec.generate.return_value = [{"text": "这是测试文本"}]
        mock_get_asr.return_value = mock_rec
        mock_load.return_value = np.zeros(16000, dtype=np.float32)

        response = self.client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.ogg", b"fake audio", "audio/ogg")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("text", data)
        self.assertEqual(data["text"], "这是测试文本")
        mock_rec.generate.assert_called()

    @patch("api_audio_transcriptions.load_audio")
    @patch("api_audio_transcriptions.get_asr_recognizer")
    def test_transcription_text_format(self, mock_get_asr, mock_load):
        mock_rec = MagicMock()
        mock_rec.generate.return_value = [{"text": "这是测试文本"}]
        mock_get_asr.return_value = mock_rec
        mock_load.return_value = np.zeros(16000, dtype=np.float32)

        response = self.client.post(
            "/v1/audio/transcriptions?response_format=text",
            files={"file": ("test.ogg", b"fake audio", "audio/ogg")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "这是测试文本")

    @patch("api_audio_transcriptions.load_audio")
    @patch("api_audio_transcriptions.get_asr_recognizer")
    def test_transcription_with_segments(self, mock_get_asr, mock_load):
        mock_rec = MagicMock()
        mock_rec.generate.return_value = [{"text": "分段文本"}]
        mock_get_asr.return_value = mock_rec
        mock_load.return_value = np.zeros(16000, dtype=np.float32)

        response = self.client.post(
            "/v1/audio/transcriptions?response_format=verbose_json",
            files={"file": ("test.ogg", b"fake audio", "audio/ogg")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("segments", data)
        self.assertEqual(len(data["segments"]), 1)
        self.assertEqual(data["segments"][0]["text"], "分段文本")

    @patch("api_audio_transcriptions.load_audio")
    @patch("api_audio_transcriptions.get_asr_recognizer")
    def test_transcription_error_handling(self, mock_get_asr, mock_load):
        mock_rec = MagicMock()
        mock_rec.generate.side_effect = Exception("测试异常")
        mock_get_asr.return_value = mock_rec
        mock_load.return_value = np.zeros(16000, dtype=np.float32)

        response = self.client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.ogg", b"fake audio", "audio/ogg")},
        )
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn("detail", data)
        self.assertEqual(data["detail"]["error"], "transcription_failed")


if __name__ == "__main__":
    unittest.main()
