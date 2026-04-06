import io
from fastapi import APIRouter, UploadFile, File
from loguru import logger
from PIL import Image
from globals import get_vision_model

router = APIRouter()


@router.post("/detection")
async def detect_image(image: UploadFile = File(...)):
    if not image:
        return {"success": False, "error": "No image found."}

    try:
        body = await image.read()
        raw = Image.open(io.BytesIO(body))
        model = get_vision_model()  # 懒加载并更新访问时间
        results = model.predict(raw, classes=0, device="gpu", verbose=False)
        predictions = []

        for result in results:
            location = result.boxes.xyxy.tolist()[0]
            label = result.names[int(result.boxes.cls.tolist()[0])]
            predictions.append(
                {
                    "x_min": location[0],
                    "y_min": location[1],
                    "x_max": location[2],
                    "y_max": location[3],
                    "confidence": result.boxes.conf.tolist()[0],
                    "label": label,
                }
            )

            if result.speed["inference"] > 70:
                logger.info(
                    f"process detection {label} time: {result.speed['inference']:0.2f}ms"
                )

        return {"predictions": predictions}
    except Exception as e:
        return {"success": False, "error": str(e)}
