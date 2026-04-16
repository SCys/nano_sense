import os
import unittest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api_embeddings import router as embeddings_router


class MockEmbeddingData:
    """模拟嵌入响应数据"""
    def __init__(self, embedding):
        self.embedding = embedding


class MockEmbeddingResponse:
    """模拟OpenAI嵌入响应"""
    def __init__(self, data, model="text-embedding-004"):
        self.data = data
        self.model = model
        self.usage = MagicMock()
        self.usage.prompt_tokens = 10
        self.usage.total_tokens = 10


class TestAPIEmbeddings(unittest.TestCase):
    """测试嵌入API"""

    @classmethod
    def setUpClass(cls):
        app = FastAPI()
        app.include_router(embeddings_router, prefix="/v1")
        cls.client = TestClient(app)

    @patch("api_embeddings.get_openai_client")
    def test_create_embeddings_single_text(self, mock_get_client):
        """测试单文本嵌入"""
        mock_client = MagicMock()
        embedding = [0.1] * 1536  # 标准维度
        mock_response = MockEmbeddingResponse([MockEmbeddingData(embedding)])
        mock_client.embeddings.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        response = self.client.post(
            "/v1/embeddings",
            json={"input": "Hello, world!", "model": "text-embedding-004"}
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("data", data)
        self.assertIsInstance(data["data"], list)
        self.assertEqual(len(data["data"]), 1)
        self.assertEqual(data["data"][0]["embedding"], embedding)
        self.assertEqual(data["model"], "text-embedding-004")
        self.assertIn("usage", data)

    @patch("api_embeddings.get_openai_client")
    def test_create_embeddings_batch(self, mock_get_client):
        """测试批量文本嵌入"""
        mock_client = MagicMock()
        embeddings = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
        mock_response = MockEmbeddingResponse(
            [MockEmbeddingData(emb) for emb in embeddings]
        )
        mock_client.embeddings.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        response = self.client.post(
            "/v1/embeddings",
            json={
                "input": ["text1", "text2", "text3"],
                "model": "text-embedding-004"
            }
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["data"]), 3)
        for i, item in enumerate(data["data"]):
            self.assertEqual(item["embedding"], embeddings[i])
            self.assertEqual(item["index"], i)

    def test_missing_input(self):
        """测试缺少输入"""
        response = self.client.post("/v1/embeddings", json={})
        self.assertEqual(response.status_code, 422)  # 验证错误

    def test_empty_input_string(self):
        """测试空字符串输入"""
        response = self.client.post(
            "/v1/embeddings",
            json={"input": ""}
        )
        self.assertEqual(response.status_code, 422)

    def test_empty_input_list(self):
        """测试空列表输入"""
        response = self.client.post(
            "/v1/embeddings",
            json={"input": []}
        )
        self.assertEqual(response.status_code, 422)

    @patch("api_embeddings.get_openai_client")
    def test_model_error(self, mock_get_client):
        """测试模型调用错误"""
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = RuntimeError(
            "API rate limit exceeded"
        )
        mock_get_client.return_value = mock_client

        response = self.client.post(
            "/v1/embeddings",
            json={"input": "test"}
        )

        self.assertEqual(response.status_code, 500)
        data = response.json()
        # FastAPI wraps HTTPException details in "detail" key
        self.assertIn("detail", data)
        detail = data["detail"]
        self.assertEqual(detail["error"], "embedding_failed")
        self.assertIn("request_id", detail)

    def test_input_exceeds_max_length(self):
        """测试超长输入"""
        long_text = "a" * 8001
        response = self.client.post(
            "/v1/embeddings",
            json={"input": long_text}
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
