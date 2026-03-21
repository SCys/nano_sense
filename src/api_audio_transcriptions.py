import datetime
import io
from datetime import datetime

import tornado
from loguru import logger

from globals import whisper_worker


class APIAudioTranscriptions(tornado.web.RequestHandler):
    def post(self):
        audio = self.request.files.get("audio")
        if not audio:
            self.write({"text": ""})
            return

        reponse_format = self.get_argument("response_format", "json")
        timestamp_granularities = self.get_argument("timestamp_granularities", None)  # word, segment

        ts_current = datetime.now()
        try:
            obj = io.BytesIO(audio[0]["body"])

            segments, info = whisper_worker.transcribe(obj, beam_size=5)

            # 打印音频信息
            logger.info(
                f"Audio duration {info.duration:.2f}s. "
                f"Detected language '{info.language}' with probability {info.language_probability}"
            )

            # 打印完整转录文本
            full_text = ""
            segments_data = []
            for segment in segments:
                start = segment.start  # 开始时间
                end = segment.end  # 结束时间
                text = segment.text  # 文本内容
                full_text += text + " "
                
                # 构建segment数据
                segment_data = {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text
                }
                segments_data.append(segment_data)
                
                logger.debug(f"[{start:.2f}s - {end:.2f}s] {text}")

            if reponse_format == "text":
                self.write(full_text.strip())
                return

            if not timestamp_granularities:
                self.write({"text": full_text.strip()})
                return

            self.write(
                {
                    "task": "transcribe",
                    "language": info.language,
                    "duration": info.duration,
                    "text": full_text.strip(),
                    "segments": segments_data,
                }
            )

        except Exception as e:
            self.write({"text": f"ASR Failed:{str(e)}"})
            logger.exception("asr failed")
        finally:
            logger.info(f"time: {datetime.now() - ts_current}")
