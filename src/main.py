import asyncio

import tornado
from loguru import logger

from api_audio_transcriptions import APIAudioTranscriptions
from api_embeddings import APIEmbeddings
from api_vision_detection import APIVisionDetection


def make_app():
    return tornado.web.Application(
        [
            (r"/v1/vision/detection", APIVisionDetection),
            (r"/v1/audio/transcriptions", APIAudioTranscriptions),
            (r"/v1/embeddings", APIEmbeddings),
        ]
    )


async def main():
    app = make_app()
    app.listen(8000)

    logger.info("Server started at port 8000")
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
