import io
import uuid
from typing import List
import torch
from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from PIL import Image
from loguru import logger
from globals import get_vision_model

router = APIRouter()


def get_request_id(request: Request) -> str:
    """从请求头获取或生成请求ID"""
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = str(uuid.uuid4())
    return request_id


@router.post("/detection")
async def detect_image(
    request: Request,
    image: UploadFile = File(...),
):
    """
    图像目标检测接口

    检测图像中的指定类别（目前仅支持类别0）。
    """
    request_id = get_request_id(request)
    log = logger.bind(request_id=request_id)

    if not image:
        raise HTTPException(status_code=400, detail="No image provided")

    try:
        body = await image.read()
        raw = Image.open(io.BytesIO(body))

        model = get_vision_model()  # 懒加载并更新访问时间
        # ultralytics 不接受 "gpu"，需用 0 (cuda:0) 或 "cpu"
        device = 0 if torch.cuda.is_available() else "cpu"
        results = model.predict(raw, classes=0, device=device, verbose=False)
        predictions: List[dict] = []

        for result in results:
            boxes_list = result.boxes.xyxy.tolist()
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

            inference_time = result.speed.get("inference", 0)
            if inference_time > 70:
                log.info(
                    f"process detection {label} time: {inference_time:0.2f}ms "
                    f"| request_id={request_id}"
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
                "message": str(e),
                "request_id": request_id,
            }
        )
