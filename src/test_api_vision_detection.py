import io
import os
import unittest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

from api_vision_detection import router as vision_router


class MockResult:
    """模拟YOLO检测结果"""
    def __init__(self):
        self.boxes = MagicMock()
        self.boxes.xyxy.tolist.return_value = [[100.0, 150.0, 300.0, 400.0]]
        self.boxes.cls.tolist.return_value = [0]
        self.boxes.conf.tolist.return_value = [0.95]
        self.names = {0: "person"}
        self.speed = {"inference": 45.2}


class TestAPIVisionDetection(unittest.TestCase):
    """测试视觉检测API"""

    @classmethod
    def setUpClass(cls):
        # 创建只包含视觉路由的FastAPI应用
        app = FastAPI()
        app.include_router(vision_router, prefix="/v1/vision")
        cls.client = TestClient(app)

    @patch("api_vision_detection.get_vision_model")
    def test_detection_success(self, mock_get_model):
        """测试成功的检测请求"""
        # 模拟YOLO模型
        mock_model = MagicMock()
        mock_model.predict.return_value = [MockResult()]
        mock_get_model.return_value = mock_model

        # 创建测试图像（1x1 像素的RGB图像）
        img = Image.new("RGB", (1, 1), color="red")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()

        response = self.client.post(
            "/v1/vision/detection",
            files={"image": ("test.png", img_bytes, "image/png")},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("predictions", data)
        self.assertIsInstance(data["predictions"], list)
        self.assertEqual(len(data["predictions"]), 1)

        pred = data["predictions"][0]
        self.assertIn("x_min", pred)
        self.assertIn("y_min", pred)
        self.assertIn("x_max", pred)
        self.assertIn("y_max", pred)
        self.assertIn("confidence", pred)
        self.assertIn("label", pred)
        self.assertEqual(pred["label"], "person")
        self.assertAlmostEqual(pred["confidence"], 0.95, places=5)

    @patch("api_vision_detection.get_vision_model")
    def test_detection_multiple_objects(self, mock_get_model):
        """测试多目标检测"""
        # 创建两个模拟结果
        mock_model = MagicMock()
        mock_model.predict.return_value = [MockResult(), MockResult()]
        mock_get_model.return_value = mock_model

        img = Image.new("RGB", (10, 10), color="blue")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes = img_bytes.getvalue()

        response = self.client.post(
            "/v1/vision/detection",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["predictions"]), 2)

    def test_no_image_provided(self):
        """测试不提供图像"""
        response = self.client.post("/v1/vision/detection")
        self.assertEqual(response.status_code, 422)  # FastAPI验证错误

    @patch("api_vision_detection.get_vision_model")
    def test_detection_model_error(self, mock_get_model):
        """测试模型内部错误"""
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("CUDA out of memory")
        mock_get_model.return_value = mock_model

        img = Image.new("RGB", (5, 5), color="green")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()

        response = self.client.post(
            "/v1/vision/detection",
            files={"image": ("test.png", img_bytes, "image/png")},
        )

        self.assertEqual(response.status_code, 500)
        data = response.json()
        # FastAPI wraps HTTPException details in "detail" key
        self.assertIn("detail", data)
        detail = data["detail"]
        self.assertEqual(detail["error"], "detection_failed")
        self.assertIn("request_id", detail)


if __name__ == "__main__":
    unittest.main()
