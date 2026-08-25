import os
import unittest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api_audio_transcriptions import router as audio_router
from config import config


class TestAPIAudioTranscriptions(unittest.TestCase):
    """测试音频转录API（端到端实际模型测试）"""

    @classmethod
    def setUpClass(cls):
        # 确保能找到真实模型路径
        asr_cfg = config.get_asr_config()
        model_path = asr_cfg["model_path"]
        if not os.path.exists(model_path):
            alt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", model_path.lstrip("./")))
            if os.path.exists(alt_path):
                os.environ["ASR_MODEL_PATH"] = alt_path

        # 确保找到 test_audio.ogg
        cls.test_audio_path = os.path.join(os.path.dirname(__file__), "assets", "test_audio.ogg")
        if not os.path.exists(cls.test_audio_path):
            cls.test_audio_path = os.path.join("assets", "test_audio.ogg")

        app = FastAPI()
        app.include_router(audio_router, prefix="/v1/audio")
        cls.client = TestClient(app)

    def test_transcription_json_format(self):
        """测试JSON格式的音频转录"""
        if not os.path.exists(self.test_audio_path):
            self.skipTest("测试音频文件不存在")

        with open(self.test_audio_path, "rb") as f:
            audio_data = f.read()

        response = self.client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test_audio.ogg", audio_data, "audio/ogg")},
        )

        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertIn("text", response_json)
        self.assertIsInstance(response_json["text"], str)

    def test_transcription_text_format(self):
        """测试TEXT格式的音频转录"""
        if not os.path.exists(self.test_audio_path):
            self.skipTest("测试音频文件不存在")

        with open(self.test_audio_path, "rb") as f:
            audio_data = f.read()

        response = self.client.post(
            "/v1/audio/transcriptions?response_format=text",
            files={"file": ("test_audio.ogg", audio_data, "audio/ogg")},
        )

        self.assertEqual(response.status_code, 200)
        response_text = response.text
        self.assertIsInstance(response_text, str)

    def test_transcription_with_segments(self):
        """测试带分段信息的音频转录"""
        if not os.path.exists(self.test_audio_path):
            self.skipTest("测试音频文件不存在")

        with open(self.test_audio_path, "rb") as f:
            audio_data = f.read()

        response = self.client.post(
            "/v1/audio/transcriptions?timestamp_granularities=segment",
            files={"file": ("test_audio.ogg", audio_data, "audio/ogg")},
        )

        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertIn("text", response_json)
        self.assertIn("segments", response_json)

    def test_empty_request(self):
        """测试空请求（不提供文件）"""
        response = self.client.post("/v1/audio/transcriptions")
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertIn("text", response_json)
        self.assertEqual(response_json["text"], "")


if __name__ == "__main__":
    unittest.main()
