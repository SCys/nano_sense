import io

import tornado
from loguru import logger
from PIL import Image

from globals import model_vision


class APIVisionDetection(tornado.web.RequestHandler):
    def post(self):
        # get image from upload
        image = self.request.files.get("image")
        if not image:
            self.write({"success": False, "error": "No image found."})
            return

        try:
            body = image[0]["body"]
            raw = Image.open(io.BytesIO(body))
            results = model_vision.predict(raw, classes=0, device="gpu", verbose=False)
            predictions = []

            # Process results list
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

                # only output slow inference > 70ms
                if result.speed["inference"] > 70:
                    logger.info(
                        f"process detection {label} time: {result.speed['inference']:0.2f}ms"
                    )

            self.write({"predictions": predictions})
        except Exception as e:
            self.write({"success": False, "error": str(e)})
