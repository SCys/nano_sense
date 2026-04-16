import threading
import time
import os
import gc
import sys
from loguru import logger
from openai import OpenAI
import ultralytics
import funasr

from config import config

class ModelManager:
    """管理模型的懒加载、超时自动卸载和重新加载"""

    def __init__(self, loader, timeout_seconds, name="model"):
        self.loader = loader
        self.timeout = timeout_seconds
        self.name = name
        self._instance = None
        self._last_access = None
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            now = time.time()
            if self._instance is None:
                logger.info(f"[{self.name}] Loading model (first access)...")
                self._instance = self.loader()
                self._last_access = now
                return self._instance
            # 检查是否超时
            if self._last_access is not None and (now - self._last_access) > self.timeout:
                logger.info(
                    f"[{self.name}] Model idle for {now - self._last_access:.1f}s, "
                    f"exceeding timeout {self.timeout}s. Unloading and reloading..."
                )
                # 显式清理资源
                self._unload_model()
                # 重新加载
                self._instance = self.loader()
                self._last_access = now
                return self._instance
            else:
                self._last_access = now
                return self._instance

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


# TTS model loader (Qwen3-TTS-ONNX)
def _load_tts_model():
    tts_config = config.get_tts_config()
    model_path = tts_config["model_path"]
    logger.info(f"Loading Qwen3-TTS model from {model_path}...")

    try:
        import onnxruntime as ort
        import numpy as np
        import soundfile as sf

        # 创建 ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        # 查找 ONNX 模型文件
        onnx_file = None
        for root, dirs, files in os.walk(model_path):
            for f in files:
                if f.endswith(".onnx"):
                    onnx_file = os.path.join(root, f)
                    break
            if onnx_file:
                break

        if not onnx_file:
            raise FileNotFoundError(f"No .onnx file found in {model_path}")

        # 创建 session（使用 CUDA）
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(onnx_file, sess_options, providers=providers)

        # 获取输入输出名称
        inputs = session.get_inputs()
        outputs = session.get_outputs()

        logger.info(f"TTS model loaded: {onnx_file}")
        logger.info(f"Inputs: {[i.name for i in inputs]}")
        logger.info(f"Outputs: {[o.name for o in outputs]}")

        return {
            "session": session,
            "inputs": {i.name: i for i in inputs},
            "outputs": {o.name: o for o in outputs},
            "model_path": model_path,
        }

    except ImportError:
        logger.warning("onnxruntime not installed, installing...")
        raise
    except Exception as e:
        logger.error(f"Failed to load TTS model: {e}")
        raise


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
    config.get_openai_config()["timeout_seconds"],
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