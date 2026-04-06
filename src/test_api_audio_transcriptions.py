import os
import unittest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api_audio_transcriptions import router as audio_router


class TestAPIAudioTranscriptions(unittest.TestCase):
    """测试音频转录API（仅音频模块）"""

    @classmethod
    def setUpClass(cls):
        # 创建仅包含音频路由的FastAPI应用，避免加载其他依赖
        app = FastAPI()
        app.include_router(audio_router, prefix="/v1/audio")
        cls.client = TestClient(app)

    def test_transcription_json_format(self):
        """测试JSON格式的音频转录"""
        test_audio_path = os.path.join("assets", "test_audio.ogg")
        self.assertTrue(os.path.exists(test_audio_path), "测试音频文件不存在")

        with open(test_audio_path, "rb") as f:
            audio_data = f.read()

        response = self.client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test_audio.ogg", audio_data, "audio/ogg")},
        )

        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertIn("text", response_json)
        self.assertIsInstance(response_json["text"], str)
        self.assertGreater(len(response_json["text"]), 0)

    def test_transcription_text_format(self):
        """测试TEXT格式的音频转录"""
        test_audio_path = os.path.join("assets", "test_audio.ogg")

        with open(test_audio_path, "rb") as f:
            audio_data = f.read()

        response = self.client.post(
            "/v1/audio/transcriptions?response_format=text",
            files={"file": ("test_audio.ogg", audio_data, "audio/ogg")},
        )

        self.assertEqual(response.status_code, 200)
        response_text = response.text
        self.assertIsInstance(response_text, str)
        self.assertGreater(len(response_text), 0)

    def test_transcription_with_segments(self):
        """测试带分段信息的音频转录"""
        test_audio_path = os.path.join("assets", "test_audio.ogg")

        with open(test_audio_path, "rb") as f:
            audio_data = f.read()

        response = self.client.post(
            "/v1/audio/transcriptions?timestamp_granularities=segment",
            files={"file": ("test_audio.ogg", audio_data, "audio/ogg")},
        )

        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertIn("text", response_json)
        self.assertIn("segments", response_json)
        self.assertIsInstance(response_json["segments"], list)
        self.assertGreater(len(response_json["segments"]), 0)

        segment = response_json["segments"][0]
        self.assertIn("id", segment)
        self.assertIn("start", segment)
        self.assertIn("end", segment)
        self.assertIn("text", segment)

    def test_empty_request(self):
        """测试空请求（不提供文件）"""
        response = self.client.post("/v1/audio/transcriptions")

        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertIn("text", response_json)
        self.assertEqual(response_json["text"], "")


if __name__ == "__main__":
    unittest.main()
