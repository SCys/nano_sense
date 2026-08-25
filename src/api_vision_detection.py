import io
import uuid
from typing import List, Tuple

import torch
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from loguru import logger
from PIL import Image

from globals import use_vision_model

router = APIRouter()

MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20MB


def get_request_id(request: Request) -> str:
    """从请求头获取或生成请求ID"""
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = str(uuid.uuid4())
    return request_id


def _sync_vision_predict(model, img_bytes: bytes) -> Tuple[List[dict], float]:
    """在工作线程池中解析图片并执行 YOLO 推理"""
    raw = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    device = 0 if torch.cuda.is_available() else "cpu"
    results = model.predict(raw, classes=0, device=device, verbose=False)
    predictions: List[dict] = []
    max_infer_time = 0.0

    for result in results:
        boxes_list = result.boxes.xyxy.tolist() if result.boxes is not None else []
        if not boxes_list:
            continue

        cls_list = result.boxes.cls.tolist()
        conf_list = result.boxes.conf.tolist()

        for i, location in enumerate(boxes_list):
            label = result.names[int(cls_list[i])]
            confidence = float(conf_list[i])

            predictions.append(
                {
                    "x_min": location[0],
                    "y_min": location[1],
                    "x_max": location[2],
                    "y_max": location[3],
                    "confidence": confidence,
                    "label": label,
                }
            )

        max_infer_time = max(max_infer_time, result.speed.get("inference", 0.0))

    return predictions, max_infer_time


@router.post("/detection")
async def detect_image(
    request: Request,
    image: UploadFile = File(...),
):
    """
    图像目标检测接口

    检测图像中的指定类别（目前仅支持类别0: person）。
    """
    request_id = get_request_id(request)
    log = logger.bind(request_id=request_id)

    if not image:
        raise HTTPException(status_code=400, detail="No image provided")

    try:
        body = await image.read()
        if not body:
            raise HTTPException(status_code=400, detail="Uploaded image is empty")

        if len(body) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"Image file exceeds maximum allowed size of {MAX_IMAGE_BYTES // 1048576}MB",
            )

        # 通过 use_vision_model 保护活跃计数，并卸载至线程池
        with use_vision_model() as model:
            predictions, inference_time = await run_in_threadpool(
                _sync_vision_predict,
                model=model,
                img_bytes=body,
            )

        if inference_time > 70:
            log.info(
                f"process detection time: {inference_time:0.2f}ms "
                f"| detections={len(predictions)} | request_id={request_id}"
            )

        return {"predictions": predictions}

    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"vision detection failed | request_id={request_id}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "detection_failed",
                "message": "Vision detection failed, please check server logs",
                "request_id": request_id,
            },
        )
