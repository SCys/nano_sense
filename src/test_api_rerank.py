import unittest
from unittest.mock import MagicMock, patch
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api_rerank import router as rerank_router


class MockRerankModel:
    """模拟 BGE Reranker 模型"""
    def __init__(self, scores=None):
        self.device = torch.device("cpu")
        self.scores = scores or [3.5, -2.0, 1.0]

    def __call__(self, **inputs):
        mock_out = MagicMock()
        mock_out.logits = torch.tensor(self.scores[: inputs["input_ids"].shape[0]], dtype=torch.float32)
        return mock_out


class MockTokenizer:
    """模拟 Tokenizer"""
    def __call__(self, pairs, **kwargs):
        n = len(pairs)
        return {
            "input_ids": torch.zeros((n, 10), dtype=torch.long),
            "attention_mask": torch.ones((n, 10), dtype=torch.long),
        }


class TestAPIRerank(unittest.TestCase):
    """测试文本重排 API (POST /v1/rerank)"""

    @classmethod
    def setUpClass(cls):
        app = FastAPI()
        app.include_router(rerank_router, prefix="/v1")
        cls.client = TestClient(app)

    @patch("api_rerank.use_rerank_model")
    def test_rerank_basic_success(self, mock_use_rerank):
        """测试基础重排与降序排序"""
        mock_tokenizer = MockTokenizer()
        # scores: doc 0=3.5 (high), doc 1=-2.0 (low), doc 2=1.0 (mid)
        mock_model = MockRerankModel(scores=[3.5, -2.0, 1.0])
        mock_use_rerank.return_value.__enter__.return_value = (mock_tokenizer, mock_model)

        response = self.client.post(
            "/v1/rerank",
            json={
                "query": "什么是人工智能？",
                "documents": [
                    "人工智能是研究让计算机模拟人类智能的科学。",
                    "今天天气真好适合去公园散步。",
                    "机器学习是人工智能的一个重要分支。",
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("id", data)
        self.assertIn("results", data)
        self.assertEqual(len(data["results"]), 3)

        # 验证降序排序：原 doc 0 (3.5) > 原 doc 2 (1.0) > 原 doc 1 (-2.0)
        results = data["results"]
        self.assertEqual(results[0]["index"], 0)
        self.assertGreater(results[0]["relevance_score"], results[1]["relevance_score"])
        self.assertEqual(results[1]["index"], 2)
        self.assertGreater(results[1]["relevance_score"], results[2]["relevance_score"])
        self.assertEqual(results[2]["index"], 1)

        # 验证包含 document
        self.assertIn("document", results[0])
        self.assertEqual(results[0]["document"]["text"], "人工智能是研究让计算机模拟人类智能的科学。")

    @patch("api_rerank.use_rerank_model")
    def test_rerank_top_n(self, mock_use_rerank):
        """测试 top_n 截断"""
        mock_tokenizer = MockTokenizer()
        mock_model = MockRerankModel(scores=[3.5, -2.0, 1.0])
        mock_use_rerank.return_value.__enter__.return_value = (mock_tokenizer, mock_model)

        response = self.client.post(
            "/v1/rerank",
            json={
                "query": "问题",
                "documents": ["doc A", "doc B", "doc C"],
                "top_n": 2,
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["results"]), 2)
        self.assertEqual(data["results"][0]["index"], 0)
        self.assertEqual(data["results"][1]["index"], 2)

    @patch("api_rerank.use_rerank_model")
    def test_rerank_without_return_documents(self, mock_use_rerank):
        """测试 return_documents=False"""
        mock_tokenizer = MockTokenizer()
        mock_model = MockRerankModel(scores=[1.0, 2.0])
        mock_use_rerank.return_value.__enter__.return_value = (mock_tokenizer, mock_model)

        response = self.client.post(
            "/v1/rerank",
            json={
                "query": "问题",
                "documents": ["doc A", "doc B"],
                "return_documents": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        for item in data["results"]:
            self.assertNotIn("document", item)
            self.assertIn("index", item)
            self.assertIn("relevance_score", item)

    @patch("api_rerank.use_rerank_model")
    def test_rerank_dict_documents(self, mock_use_rerank):
        """测试字典格式的候选文档输入"""
        mock_tokenizer = MockTokenizer()
        mock_model = MockRerankModel(scores=[2.0, 0.5])
        mock_use_rerank.return_value.__enter__.return_value = (mock_tokenizer, mock_model)

        docs = [
            {"id": "doc-101", "text": "这是字典格式的文档A", "meta": "info"},
            {"id": "doc-102", "text": "这是字典格式的文档B", "meta": "info"},
        ]

        response = self.client.post(
            "/v1/rerank",
            json={
                "query": "文档A",
                "documents": docs,
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["results"][0]["document"]["id"], "doc-101")

    def test_rerank_empty_query(self):
        """测试空 query 抛出 422 验证错误"""
        response = self.client.post(
            "/v1/rerank",
            json={"query": "   ", "documents": ["doc"]},
        )
        self.assertEqual(response.status_code, 422)

    def test_rerank_empty_documents(self):
        """测试空 documents 抛出 422 验证错误"""
        response = self.client.post(
            "/v1/rerank",
            json={"query": "query", "documents": []},
        )
        self.assertEqual(response.status_code, 422)

    @patch("api_rerank.use_rerank_model")
    def test_rerank_model_error(self, mock_use_rerank):
        """测试模型推理异常捕获与脱敏"""
        mock_use_rerank.side_effect = RuntimeError("CUDA internal error")

        response = self.client.post(
            "/v1/rerank",
            json={"query": "query", "documents": ["doc"]},
        )

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertEqual(data["detail"]["error"], "rerank_failed")
        self.assertIn("request_id", data["detail"])


if __name__ == "__main__":
    unittest.main()
