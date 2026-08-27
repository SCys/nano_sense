import os
from loguru import logger

# 尝试使用 tomllib (Python 3.11+), 否则使用 toml 包
try:
    import tomllib

    HAS_TOMLLIB = True
except ImportError:
    HAS_TOMLLIB = False
    try:
        import toml

        HAS_TOML = True
    except ImportError:
        HAS_TOML = False


class Config:
    """配置加载类，负责从 TOML 文件加载配置，并支持环境变量覆盖"""

    def __init__(self, config_file="main.toml"):
        # 检查配置文件是否存在
        if not os.path.exists(config_file):
            logger.error(f"配置文件 {config_file} 不存在")
            raise FileNotFoundError(f"配置文件 {config_file} 不存在")

        # 读取配置
        with open(
            config_file,
            "rb" if HAS_TOMLLIB else "r",
            encoding=None if HAS_TOMLLIB else "utf-8",
        ) as f:
            if HAS_TOMLLIB:
                self.config = tomllib.load(f)
            elif HAS_TOML:
                self.config = toml.load(f)
            else:
                raise RuntimeError("未找到 toml 解析器，请安装 Python 3.11+ 或 toml 包")

        logger.info(f"成功加载配置文件: {config_file}")

    def _get_env_or_config(self, section, option, env_var, fallback=None):
        """优先从环境变量获取，其次从配置文件获取"""
        env_value = os.getenv(env_var)
        if env_value is not None:
            return env_value
        return self.config.get(section, {}).get(option, fallback)

    def get(self, section, option, fallback=None):
        """获取配置项"""
        return self.config.get(section, {}).get(option, fallback)

    def getint(self, section, option, fallback=None):
        """获取整数配置项"""
        value = self.get(section, option, fallback)
        if value is None or value == fallback:
            return fallback
        return int(value)

    def getfloat(self, section, option, fallback=None):
        """获取浮点数配置项"""
        value = self.get(section, option, fallback)
        if value is None or value == fallback:
            return fallback
        return float(value)

    def getboolean(self, section, option, fallback=None):
        """获取布尔值配置项（正确处理字符串形式的布尔值）"""
        value = self.get(section, option, fallback)
        if value is None or value == fallback:
            return fallback
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def get_asr_config(self):
        """获取 ASR 配置（支持环境变量）"""
        model_path = self._get_env_or_config(
            "asr", "model_path", "ASR_MODEL_PATH",
            fallback="/app/models/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        )
        timeout_seconds = self.getint("asr", "timeout_seconds", fallback=1800)

        # VAD（长音频切分）与标点恢复模型；设为空字符串可关闭
        vad_model = self.get("asr", "vad_model", fallback="fsmn-vad")
        punc_model = self.get("asr", "punc_model", fallback="ct-punc")

        # 推理设备：cuda:0 / cuda:1 / cpu；默认自动选第一块可见卡
        device = self._get_env_or_config("asr", "device", "ASR_DEVICE",
                                         fallback="cuda:0")

        if not os.path.exists(model_path):
            logger.warning(f"ASR model path does not exist: {model_path}")

        return {
            "timeout_seconds": timeout_seconds,
            "model_path": model_path,
            "vad_model": vad_model or None,
            "punc_model": punc_model or None,
            "device": device,
        }

    def get_ultralytics_config(self):
        """获取Ultralytics配置（支持环境变量）"""
        model_path = self._get_env_or_config(
            "ultralytics", "model_path", "ULTRALYTICS_MODEL_PATH"
        )
        timeout_seconds = self.getint(
            "ultralytics", "timeout_seconds", fallback=1800
        )

        if not model_path:
            raise ValueError(
                "Ultralytics model path is required. Set it in main.toml "
                "or via ULTRALYTICS_MODEL_PATH environment variable."
            )

        if not os.path.exists(model_path):
            logger.warning(f"Ultralytics model path does not exist: {model_path}")

        return {
            "model_path": model_path,
            "timeout_seconds": timeout_seconds,
        }

    def get_openai_config(self):
        """获取OpenAI配置（支持环境变量）"""
        api_key = self._get_env_or_config(
            "openai", "api_key", "OPENAI_API_KEY"
        )
        base_url = self._get_env_or_config(
            "openai", "base_url", "OPENAI_BASE_URL"
        )
        timeout_seconds = self.getint(
            "openai", "timeout_seconds", fallback=1800
        )

        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set it in main.toml "
                "or via OPENAI_API_KEY environment variable."
            )

        return {
            "api_key": api_key,
            "base_url": base_url,
            "timeout_seconds": timeout_seconds,
        }

    def get_tts_config(self):
        """获取TTS配置（支持环境变量）"""
        model_path = self._get_env_or_config(
            "tts", "model_path", "TTS_MODEL_PATH",
            fallback="./data/openbmb/VoxCPM2",
        )
        timeout_seconds = self.getint("tts", "timeout_seconds", fallback=1800)
        device = self._get_env_or_config(
            "tts", "device", "TTS_DEVICE",
            fallback="cuda:0",
        )

        if not os.path.exists(model_path):
            logger.warning(f"TTS model path does not exist: {model_path}")

        return {
            "model_path": model_path,
            "timeout_seconds": timeout_seconds,
            "device": device,
        }

    def get_rerank_config(self):
        """获取 Rerank 重排模型配置（支持环境变量）"""
        model_path = self._get_env_or_config(
            "rerank", "model_path", "RERANK_MODEL_PATH",
            fallback="./data/BAAI/bge-reranker-base",
        )
        timeout_seconds = self.getint("rerank", "timeout_seconds", fallback=1800)
        device = self._get_env_or_config(
            "rerank", "device", "RERANK_DEVICE",
            fallback="cuda:1",
        )

        if not os.path.exists(model_path):
            logger.warning(f"Rerank model path does not exist: {model_path}")

        return {
            "model_path": model_path,
            "timeout_seconds": timeout_seconds,
            "device": device,
        }


# 创建全局配置实例
try:
    config = Config()
except Exception as e:
    logger.error(f"加载配置失败: {str(e)}")
    raise
