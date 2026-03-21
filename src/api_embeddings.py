import datetime
from datetime import datetime

import orjson as json
import tornado
from loguru import logger

from globals import openai_client


class APIEmbeddings(tornado.web.RequestHandler):
    """
    curl "https://generativelanguage.googleapis.com/v1beta/openai/embeddings" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer GEMINI_API_KEY" \
    -d '{
        "input": "Your text string goes here",
        "model": "text-embedding-004"
    }'
    """

    def post(self):
        data = json.loads(self.request.body)
        input = data.get("input")
        model = data.get("model", "text-embedding-004")

        if not input:
            self.write({"error": "input is required"})
            return

        ts_current = datetime.now()
        try:
            response = openai_client.embeddings.create(
                input=input,
                model=model,
            )

            logger.info(f"embedding time: {datetime.now() - ts_current}")

            self.write({"object": "embedding", "embedding": response.data[0].embedding, "index": 0})
        except Exception as e:
            logger.exception("embedding failed")
            self.write({"error": str(e)})
