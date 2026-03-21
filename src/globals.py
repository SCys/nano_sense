import datetime
import threading
import time
from datetime import datetime

import ultralytics
from loguru import logger
from openai import OpenAI

from config import config

# 从配置文件获取Whisper配置
whisper_config = config.get_whisper_config()
WHISPER_MODEL = whisper_config["model"]
WHISPER_TIMEOUT_SECONDS = whisper_config["timeout_seconds"]

class WhisperWorker:
    def __init__(self, timeout=WHISPER_TIMEOUT_SECONDS):
        self.timeout = timeout
        self._model = None
        self._last_access = None
        self._lock = threading.Lock()
        self._supervisor_thread = threading.Thread(target=self._supervise, daemon=True)
        self._supervisor_thread.start()

    def _load_model(self):
        ts_current = datetime.now()
        self._model = WhisperModel(
            WHISPER_MODEL,
            device=whisper_config["device"],
            download_root=whisper_config["download_root"],
        )
        logger.info(f"Audio model loaded. time: {datetime.now() - ts_current}")

    def get_model(self):
        with self._lock:
            if self._model is None:
                self._load_model()
            self._last_access = time.time()
            return self._model
            
    def transcribe(self, audio_file, beam_size=5):
        """
        转录音频文件
        
        Args:
            audio_file: 音频文件对象
            beam_size: beam search大小
            
        Returns:
            segments: 分段转录结果
            info: 音频信息
        """
        model = self.get_model()
        segments, info = model.transcribe(
            audio_file, 
            beam_size=beam_size
        )
        self._last_access = time.time()
        return segments, info

    def _unload_model(self):
        with self._lock:
            if self._model is not None:
                logger.info("Unloading Whisper model from memory due to inactivity.")
                self._model = None

    def _supervise(self):
        while True:
            time.sleep(60)
            with self._lock:
                if self._model is not None and self._last_access is not None:
                    if time.time() - self._last_access > self.timeout:
                        self._unload_model()

def setup_ultralytics():
    # 从配置获取YOLO模型路径
    ultralytics_config = config.get_ultralytics_config()
    
    ultralytics.checks(verbose=False)
    model = ultralytics.YOLO(ultralytics_config["model_path"], verbose=True)

    logger.info("Ultraytics model loaded.")

    return model


def setup_openai_client():
    # 从配置获取OpenAI配置
    openai_config = config.get_openai_config()
    
    client = OpenAI(
        api_key=openai_config["api_key"],
        base_url=openai_config["base_url"],
    )

    logger.info("OpenAI client loaded.")

    return client


model_vision = setup_ultralytics()
whisper_worker = WhisperWorker()
openai_client = setup_openai_client()
