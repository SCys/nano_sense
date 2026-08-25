import threading
import time
import os
import gc
import json
from loguru import logger
from openai import OpenAI
import ultralytics
import funasr

from config import config

# 所有 ModelManager 实例注册表，供看门狗扫描
_MODEL_MANAGERS = []
_WATCHDOG_STARTED = False

class ModelManager:
    """管理模型的懒加载、超时自动下线和重新加载。

    - get()：首次访问时加载；已加载则刷新访问时间并返回。
    - 空闲超时由后台看门狗线程主动卸载（见 _start_unload_watchdog），
      下次 get() 时重新加载。
    - 注意：get() 持锁执行加载，同一模型的并发请求会串行等待；
      这是刻意设计——避免大模型被并发重复加载撑爆显存。
    """

    def __init__(self, loader, timeout_seconds, name="model"):
        self.loader = loader
        self.timeout = timeout_seconds
        self.name = name
        self._instance = None
        self._last_access = None
        self._lock = threading.Lock()
        _MODEL_MANAGERS.append(self)

    @property
    def loaded(self):
        return self._instance is not None

    def get(self):
        with self._lock:
            if self._instance is None:
                logger.info(f"[{self.name}] Loading model (first access)...")
                self._instance = self.loader()
            self._last_access = time.time()
            return self._instance

    def unload_if_idle(self):
        """若空闲时间超过 timeout 则卸载模型。

        由看门狗线程周期性调用；返回是否执行了卸载。
        卸载后下次 get() 会重新加载。
        """
        with self._lock:
            if self._instance is None or self._last_access is None:
                return False
            idle = time.time() - self._last_access
            if idle >= self.timeout:
                logger.info(
                    f"[{self.name}] Idle for {idle:.0f}s >= timeout {self.timeout}s, "
                    f"auto-unloading..."
                )
                self._unload_model()
                return True
            return False

    def _unload_model(self):
        """卸载模型并清理资源"""
        import torch

        if self._instance is not None:
            # 记录模型信息用于调试
            model_type = type(self._instance).__name__
            logger.info(f"[{self.name}] Unloading {model_type} instance...")

            # 删除引用
            del self._instance
            self._instance = None

            # 强制垃圾回收
            gc.collect()

            # 清理 CUDA 缓存（如果可用）
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.debug(f"[{self.name}] CUDA cache cleared")

        logger.info(f"[{self.name}] Model unloaded successfully")

def _load_funasr_recognizer():
    """
    from https://github.com/modelscope/FunASR/blob/main/README_zh.md

    挂载 fsmn-vad（长音频切分）与 ct-punc（中文标点恢复/断句），
    可在 main.toml [asr] 里用 vad_model="" / punc_model="" 关闭。
    """
    asr_config = config.get_asr_config()

    model_path = asr_config["model_path"]
    kwargs = {
        "model": model_path,
        "device": asr_config.get("device", "cuda:0"),
        "disable_update": True,
    }
    if asr_config.get("vad_model"):
        kwargs["vad_model"] = asr_config["vad_model"]
        kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}
    if asr_config.get("punc_model"):
        kwargs["punc_model"] = asr_config["punc_model"]

    logger.info(f"Loading funasr recognizer from {model_path} "
                f"(vad={asr_config.get('vad_model')}, punc={asr_config.get('punc_model')})...")
    return funasr.AutoModel(**kwargs)

# Vision model loader
def _load_vision_model():
    ultralytics_config = config.get_ultralytics_config()
    model_path = ultralytics_config["model_path"]
    logger.info(f"Loading ultralytics YOLO model from {model_path}...")
    ultralytics.checks(verbose=False)
    return ultralytics.YOLO(model_path, verbose=True)

# OpenAI client loader
def _load_openai_client():
    # 延迟到首次调用才校验 api_key，避免未使用 embeddings 的部署因缺少 key 无法启动
    openai_config = config.get_openai_config()
    logger.info("Loading OpenAI client...")
    return OpenAI(
        api_key=openai_config["api_key"],
        base_url=openai_config["base_url"],
    )


# TTS model loader (VoxCPM2)
def _load_tts_model():
    """加载 OpenBMB VoxCPM2 模型。

    支持 30 种语言、48kHz 高采样率、声音设计（文本描述）与声音克隆（参考音频）。
    """
    tts_config = config.get_tts_config()
    model_path = tts_config["model_path"]
    device = tts_config.get("device", "cuda:0")
    logger.info(f"Loading VoxCPM2 model from {model_path} on {device}...")

    from voxcpm import VoxCPM
    model = VoxCPM.from_pretrained(
        model_path,
        load_denoiser=False,
        device=device,
    )
    logger.info(f"VoxCPM2 loaded successfully on {device} (sample_rate={model.tts_model.sample_rate})")
    return model


# 创建 ModelManager 实例（带名称用于日志追踪）
asr_manager = ModelManager(
    _load_funasr_recognizer,
    config.get_asr_config()["timeout_seconds"],
    name="ASR"
)
vision_manager = ModelManager(
    _load_vision_model,
    config.get_ultralytics_config()["timeout_seconds"],
    name="Vision"
)
openai_manager = ModelManager(
    _load_openai_client,
    config.getint("openai", "timeout_seconds", fallback=300),
    name="OpenAI"
)
tts_manager = ModelManager(
    _load_tts_model,
    config.get_tts_config()["timeout_seconds"],
    name="TTS"
)

# 提供获取模型的函数
def get_asr_recognizer():
    return asr_manager.get()

def get_vision_model():
    return vision_manager.get()

def get_openai_client():
    return openai_manager.get()

def get_tts_model():
    return tts_manager.get()


def _start_unload_watchdog(interval_seconds: int = 60):
    """启动后台看门狗：周期性检查各模型空闲时长，超时则自动下线。

    - interval_seconds：扫描间隔，默认 60s（卸载精度受此间隔影响）
    - daemon 线程，进程退出时自动结束，不阻塞服务关闭
    """
    global _WATCHDOG_STARTED
    if _WATCHDOG_STARTED:
        return
    _WATCHDOG_STARTED = True

    def _loop():
        while True:
            time.sleep(interval_seconds)
            for manager in list(_MODEL_MANAGERS):
                try:
                    if manager.unload_if_idle():
                        logger.info(f"[{manager.name}] Model offline, will reload on next request")
                except Exception as e:
                    logger.warning(f"[{manager.name}] Unload watchdog error: {e}")

    watchdog = threading.Thread(target=_loop, name="model-unload-watchdog", daemon=True)
    watchdog.start()
    logger.info(
        f"Model unload watchdog started (interval={interval_seconds}s, "
        f"managers={[m.name for m in _MODEL_MANAGERS]})"
    )


_start_unload_watchdog(60)