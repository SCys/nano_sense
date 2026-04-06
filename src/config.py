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
    """配置加载类，负责从 TOML 文件加载配置"""

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
        """获取布尔值配置项"""
        value = self.get(section, option, fallback)
        if value is None or value == fallback:
            return fallback
        return bool(value)

    def get_asr_config(self):
        """获取 ASR 配置"""
        return {
            "timeout_seconds": self.getint("asr", "timeout_seconds", fallback=300),
            "model_path": self.get(
                "asr",
                "model_path",
                fallback="/app/models/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            ),
        }

    def get_ultralytics_config(self):
        """获取Ultralytics配置"""
        return {
            "model_path": self.get("ultralytics", "model_path"),
            "timeout_seconds": self.getint(
                "ultralytics", "timeout_seconds", fallback=300
            ),
        }

    def get_openai_config(self):
        """获取OpenAI配置"""
        return {
            "api_key": self.get("openai", "api_key"),
            "base_url": self.get("openai", "base_url"),
            "timeout_seconds": self.getint("openai", "timeout_seconds", fallback=300),
        }


# 创建全局配置实例
try:
    config = Config()
except Exception as e:
    logger.error(f"加载配置失败: {str(e)}")
    raise
