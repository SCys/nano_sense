import gc
import json
import os
import threading
import time
from contextlib import contextmanager
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

    - get() / use()：首次访问时加载；已加载则刷新访问时间并返回。
    - use() 上下文管理器：自动维护活跃引用计数，防止看门狗在推理执行中强行卸载导致 CUDA 崩溃。
    - 空闲超时由后台看门狗线程主动卸载（仅在 active_count == 0 且空闲超时时卸载）。
    - 注意：加载过程持锁执行，避免并发重复加载撑爆显存。
    """

    def __init__(self, loader, timeout_seconds, name="model"):
        self.loader = loader
        self.timeout = timeout_seconds
        self.name = name
        self._instance = None
        self._last_access = None
        self._active_count = 0
        self._lock = threading.Lock()
        _MODEL_MANAGERS.append(self)

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._instance is not None

    @property
    def active_count(self) -> int:
        with self._lock:
            return self._active_count

    def get(self):
        """获取模型实例并刷新访问时间"""
        with self._lock:
            if self._instance is None:
                logger.info(f"[{self.name}] Loading model (first access)...")
                try:
                    self._instance = self.loader()
                except Exception:
                    self._instance = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise
            self._last_access = time.time()
            return self._instance

    @contextmanager
    def use(self):
        """上下文管理器：安全借用模型并累加活跃引用计数，阻止运行中被卸载"""
        with self._lock:
            if self._instance is None:
                logger.info(f"[{self.name}] Loading model (first access)...")
                try:
                    self._instance = self.loader()
                except Exception:
                    self._instance = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise
            self._active_count += 1
            self._last_access = time.time()
            instance = self._instance

        try:
            yield instance
        finally:
            with self._lock:
                self._active_count = max(0, self._active_count - 1)
                self._last_access = time.time()

    def unload_if_idle(self) -> bool:
        """若空闲时间超过 timeout 且当前无活跃推理，则卸载模型。

        由看门狗线程周期性调用；返回是否执行了卸载。
        """
        with self._lock:
            if self._instance is None or self._last_access is None:
                return False
            if self._active_count > 0:
                logger.debug(f"[{self.name}] In active use (active_count={self._active_count}), skip unload")
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

    logger.info(
        f"Loading funasr recognizer from {model_path} "
        f"(vad={asr_config.get('vad_model')}, punc={asr_config.get('punc_model')}, device={kwargs['device']})..."
    )
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
    若 GPU 显存暂时被宿主机其他大型进程占满（OOM），自动清理显存并降级至 CPU 运行。
    """
    tts_config = config.get_tts_config()
    model_path = tts_config["model_path"]
    device = tts_config.get("device", "cuda:0")
    logger.info(f"Loading VoxCPM2 model from {model_path} on {device}...")

    import torch
    from voxcpm import VoxCPM

    try:
        model = VoxCPM.from_pretrained(
            model_path,
            load_denoiser=False,
            optimize=False,
            device=device,
        )
        logger.info(f"VoxCPM2 loaded successfully on {device} (sample_rate={model.tts_model.sample_rate})")
        return model
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            logger.warning(
                f"GPU {device} out of memory when loading VoxCPM2 ({e}), "
                f"clearing CUDA cache and falling back to CPU..."
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = VoxCPM.from_pretrained(
                model_path,
                load_denoiser=False,
                optimize=False,
                device="cpu",
            )
            logger.info(f"VoxCPM2 loaded successfully on CPU (fallback mode, sample_rate={model.tts_model.sample_rate})")
            return model
        raise


# 创建 ModelManager 实例（带名称用于日志追踪）
asr_manager = ModelManager(
    _load_funasr_recognizer,
    config.get_asr_config()["timeout_seconds"],
    name="ASR",
)
vision_manager = ModelManager(
    _load_vision_model,
    config.get_ultralytics_config()["timeout_seconds"],
    name="Vision",
)
openai_manager = ModelManager(
    _load_openai_client,
    config.getint("openai", "timeout_seconds", fallback=1800),
    name="OpenAI",
)
tts_manager = ModelManager(
    _load_tts_model,
    config.get_tts_config()["timeout_seconds"],
    name="TTS",
)


# 提供获取模型的函数与上下文管理器
def get_asr_recognizer():
    return asr_manager.get()


def use_asr_recognizer():
    return asr_manager.use()


def get_vision_model():
    return vision_manager.get()


def use_vision_model():
    return vision_manager.use()


def get_openai_client():
    return openai_manager.get()


def use_openai_client():
    return openai_manager.use()


def get_tts_model():
    return tts_manager.get()


def use_tts_model():
    return tts_manager.use()


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
