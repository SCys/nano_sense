import threading
import time
import os
from loguru import logger
from openai import OpenAI
import ultralytics
import funasr

from config import config


class ModelManager:
    """管理模型的懒加载、超时自动卸载和重新加载"""

    def __init__(self, loader, timeout_seconds):
        self.loader = loader
        self.timeout = timeout_seconds
        self._instance = None
        self._last_access = None
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            now = time.time()
            if self._instance is None:
                logger.info(f"Loading model (first access)...")
                self._instance = self.loader()
                self._last_access = now
                return self._instance
            # 检查是否超时
            if self._last_access is not None and (now - self._last_access) > self.timeout:
                logger.info(
                    f"Model idle for {now - self._last_access:.1f}s, exceeding timeout {self.timeout}s. Unloading and reloading..."
                )
                # 卸载旧实例
                del self._instance
                self._instance = None
                self._last_access = None
                # 重新加载
                self._instance = self.loader()
                self._last_access = now
                return self._instance
            else:
                self._last_access = now
                return self._instance


def _load_funasr_recognizer():
    """
    from https://github.com/modelscope/FunASR/blob/main/README_zh.md
    """
    asr_config = config.get_asr_config()

    model_path = asr_config["model_path"]
    logger.info(f"Loading funasr recognizer from {model_path}...")
    return funasr.AutoModel(
        model=model_path,
        device="cuda:0",
        disable_update=True,
    )

# Vision model loader
def _load_vision_model():
    ultralytics_config = config.get_ultralytics_config()
    model_path = ultralytics_config["model_path"]
    logger.info(f"Loading ultralytics YOLO model from {model_path}...")
    ultralytics.checks(verbose=False)
    return ultralytics.YOLO(model_path, verbose=True)


# OpenAI client loader
def _load_openai_client():
    openai_config = config.get_openai_config()
    logger.info("Loading OpenAI client...")
    return OpenAI(
        api_key=openai_config["api_key"],
        base_url=openai_config["base_url"],
    )


# 创建 ModelManager 实例
asr_manager = ModelManager(_load_funasr_recognizer, config.get_asr_config()["timeout_seconds"])
vision_manager = ModelManager(_load_vision_model, config.get_ultralytics_config()["timeout_seconds"])
openai_manager = ModelManager(_load_openai_client, config.get_openai_config()["timeout_seconds"])


# 提供获取模型的函数
def get_asr_recognizer():
    return asr_manager.get()


def get_vision_model():
    return vision_manager.get()


def get_openai_client():
    return openai_manager.get()
