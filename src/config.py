import configparser
import os
from loguru import logger

class Config:
    """配置加载类，负责从main.ini加载配置"""
    
    def __init__(self, config_file='main.ini'):
        self.config = configparser.ConfigParser()
        
        # 检查配置文件是否存在
        if not os.path.exists(config_file):
            logger.error(f"配置文件 {config_file} 不存在")
            raise FileNotFoundError(f"配置文件 {config_file} 不存在")
            
        # 加载配置文件
        self.config.read(config_file, encoding='utf-8')
        logger.info(f"成功加载配置文件: {config_file}")
        
    def get(self, section, option, fallback=None):
        """获取配置项"""
        return self.config.get(section, option, fallback=fallback)
        
    def getint(self, section, option, fallback=None):
        """获取整数配置项"""
        return self.config.getint(section, option, fallback=fallback)
        
    def getfloat(self, section, option, fallback=None):
        """获取浮点数配置项"""
        return self.config.getfloat(section, option, fallback=fallback)
        
    def getboolean(self, section, option, fallback=None):
        """获取布尔值配置项"""
        return self.config.getboolean(section, option, fallback=fallback)
    
    def get_whisper_config(self):
        """获取Whisper配置"""
        return {
            "model": self.get("whisper", "model"),
            "device": self.get("whisper", "device"),
            "download_root": self.get("whisper", "download_root"),
            "timeout_seconds": self.getint("whisper", "timeout_seconds")
        }
    
    def get_ultralytics_config(self):
        """获取Ultralytics配置"""
        return {
            "model_path": self.get("ultralytics", "model_path")
        }
    
    def get_openai_config(self):
        """获取OpenAI配置"""
        return {
            "api_key": self.get("openai", "api_key"),
            "base_url": self.get("openai", "base_url")
        }

# 创建全局配置实例
try:
    config = Config()
except Exception as e:
    logger.error(f"加载配置失败: {str(e)}")
    raise 