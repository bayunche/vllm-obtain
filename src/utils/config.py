"""
配置管理系统
支持环境变量、.env文件和配置验证
"""

import os
from pathlib import Path
from typing import Optional, List, Any, Dict
from pydantic import BaseSettings, Field, validator
from dotenv import load_dotenv
from loguru import logger


class InferenceConfig(BaseSettings):
    """推理服务配置类"""
    
    # 服务配置
    host: str = Field(default="0.0.0.0", description="服务监听地址")
    port: int = Field(default=8000, description="服务端口")
    workers: int = Field(default=4, description="Worker进程数")
    debug: bool = Field(default=False, description="调试模式")
    
    # 推理引擎配置
    inference_engine: str = Field(default="auto", description="推理引擎选择")
    inference_mode: str = Field(default="single", description="推理模式")
    
    # 模型配置
    max_concurrent_models: int = Field(default=1, description="最大并发模型数")
    default_model: str = Field(default="Qwen2.5-7B-Instruct", description="默认模型")
    model_base_path: str = Field(default="./models", description="模型基础路径")
    model_cache_dir: str = Field(default="./cache", description="模型缓存目录")
    
    # 硬件配置
    device_type: str = Field(default="auto", description="设备类型")
    max_gpu_memory: float = Field(default=0.8, description="最大GPU内存使用率")
    max_cpu_threads: int = Field(default=8, description="最大CPU线程数")
    
    # OpenAI API 兼容配置
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API密钥")
    openai_api_base: str = Field(default="https://api.openai.com/v1", description="OpenAI API基础URL")
    
    # 日志配置
    log_level: str = Field(default="INFO", description="日志级别")
    log_file: str = Field(default="./logs/inference.log", description="日志文件路径")
    log_rotation: str = Field(default="100MB", description="日志轮转大小")
    log_retention: str = Field(default="7 days", description="日志保留时间")
    
    # 负载均衡配置
    cluster_instances: int = Field(default=1, description="集群实例数量")
    cluster_nodes: Optional[str] = Field(default=None, description="集群节点列表")
    load_balance_strategy: str = Field(default="round_robin", description="负载均衡策略")
    health_check_interval: int = Field(default=30, description="健康检查间隔(秒)")
    health_check_timeout: int = Field(default=5, description="健康检查超时(秒)")
    auto_scaling: bool = Field(default=False, description="启用自动扩缩容")
    min_instances: int = Field(default=1, description="最小实例数")
    max_instances: int = Field(default=10, description="最大实例数")
    
    # 性能优化配置
    enable_caching: bool = Field(default=True, description="启用缓存")
    cache_size: int = Field(default=1000, description="缓存大小")
    request_timeout: int = Field(default=300, description="请求超时时间(秒)")
    max_concurrent_requests: int = Field(default=100, description="最大并发请求数")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @validator("inference_engine")
    def validate_inference_engine(cls, v):
        """验证推理引擎选择"""
        valid_engines = ["auto", "vllm", "mlx", "llama_cpp"]
        if v not in valid_engines:
            raise ValueError(f"推理引擎必须是 {valid_engines} 中的一个")
        return v
    
    @validator("inference_mode")
    def validate_inference_mode(cls, v):
        """验证推理模式"""
        valid_modes = ["single", "multi_instance", "load_balance"]
        if v not in valid_modes:
            raise ValueError(f"推理模式必须是 {valid_modes} 中的一个")
        return v
    
    @validator("load_balance_strategy")
    def validate_load_balance_strategy(cls, v):
        """验证负载均衡策略"""
        valid_strategies = ["round_robin", "least_connections", "weighted", "random", "response_time"]
        if v not in valid_strategies:
            raise ValueError(f"负载均衡策略必须是 {valid_strategies} 中的一个")
        return v
    
    @validator("device_type")
    def validate_device_type(cls, v):
        """验证设备类型"""
        valid_devices = ["auto", "cuda", "mps", "cpu"]
        if v not in valid_devices:
            raise ValueError(f"设备类型必须是 {valid_devices} 中的一个")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """验证日志级别"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"日志级别必须是 {valid_levels} 中的一个")
        return v.upper()
    
    @validator("max_gpu_memory")
    def validate_max_gpu_memory(cls, v):
        """验证GPU内存使用率"""
        if not 0.1 <= v <= 1.0:
            raise ValueError("GPU内存使用率必须在0.1到1.0之间")
        return v
    
    @validator("max_concurrent_models")
    def validate_max_concurrent_models(cls, v):
        """验证最大并发模型数"""
        if v < 1:
            raise ValueError("最大并发模型数必须大于0")
        return v
    
    @validator("port")
    def validate_port(cls, v):
        """验证端口号"""
        if not 1 <= v <= 65535:
            raise ValueError("端口号必须在1到65535之间")
        return v
    
    def get_cluster_nodes(self) -> List[str]:
        """获取集群节点列表"""
        if not self.cluster_nodes:
            return []
        return [node.strip() for node in self.cluster_nodes.split(",")]
    
    def create_directories(self):
        """创建必要的目录"""
        directories = [
            self.model_base_path,
            self.model_cache_dir,
            os.path.dirname(self.log_file)
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.debug(f"确保目录存在: {directory}")


class ConfigManager:
    """配置管理器"""
    
    _instance: Optional[InferenceConfig] = None
    _config_file_path: Optional[str] = None
    
    @classmethod
    def load_config(cls, config_file: Optional[str] = None) -> InferenceConfig:
        """
        加载配置
        
        Args:
            config_file: 配置文件路径，默认为.env
            
        Returns:
            配置对象
        """
        if cls._instance is None or config_file != cls._config_file_path:
            cls._config_file_path = config_file
            
            # 加载环境变量文件
            if config_file:
                if os.path.exists(config_file):
                    load_dotenv(config_file)
                    logger.info(f"已加载配置文件: {config_file}")
                else:
                    logger.warning(f"配置文件不存在: {config_file}")
            else:
                # 尝试加载默认的.env文件
                env_files = [".env", ".env.local"]
                for env_file in env_files:
                    if os.path.exists(env_file):
                        load_dotenv(env_file)
                        logger.info(f"已加载默认配置文件: {env_file}")
                        break
            
            # 创建配置实例
            try:
                cls._instance = InferenceConfig()
                cls._instance.create_directories()
                logger.info("配置加载成功")
                cls._log_config_summary(cls._instance)
            except Exception as e:
                logger.error(f"配置加载失败: {e}")
                raise
        
        return cls._instance
    
    @classmethod
    def get_config(cls) -> InferenceConfig:
        """获取当前配置"""
        if cls._instance is None:
            return cls.load_config()
        return cls._instance
    
    @classmethod
    def reload_config(cls, config_file: Optional[str] = None) -> InferenceConfig:
        """重新加载配置"""
        cls._instance = None
        return cls.load_config(config_file)
    
    @classmethod
    def _log_config_summary(cls, config: InferenceConfig):
        """记录配置摘要"""
        logger.info("=== 配置摘要 ===")
        logger.info(f"服务地址: {config.host}:{config.port}")
        logger.info(f"推理引擎: {config.inference_engine}")
        logger.info(f"推理模式: {config.inference_mode}")
        logger.info(f"设备类型: {config.device_type}")
        logger.info(f"默认模型: {config.default_model}")
        logger.info(f"最大并发模型: {config.max_concurrent_models}")
        logger.info(f"日志级别: {config.log_level}")
        
        if config.inference_mode == "load_balance":
            nodes = config.get_cluster_nodes()
            logger.info(f"集群节点数: {len(nodes)}")
    
    @classmethod
    def validate_environment(cls) -> Dict[str, Any]:
        """验证运行环境"""
        validation_result = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        config = cls.get_config()
        
        # 检查模型路径
        model_path = Path(config.model_base_path)
        if not model_path.exists():
            validation_result["warnings"].append(f"模型路径不存在: {model_path}")
        
        # 检查缓存目录权限
        cache_path = Path(config.model_cache_dir)
        try:
            cache_path.mkdir(parents=True, exist_ok=True)
            test_file = cache_path / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            validation_result["errors"].append(f"缓存目录无写入权限: {e}")
            validation_result["valid"] = False
        
        # 检查日志目录权限
        log_path = Path(config.log_file).parent
        try:
            log_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            validation_result["errors"].append(f"日志目录创建失败: {e}")
            validation_result["valid"] = False
        
        # 检查端口可用性
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((config.host, config.port))
        except OSError as e:
            validation_result["errors"].append(f"端口 {config.port} 不可用: {e}")
            validation_result["valid"] = False
        
        return validation_result


# 全局配置实例
def get_config() -> InferenceConfig:
    """获取全局配置实例"""
    return ConfigManager.get_config()


if __name__ == "__main__":
    # 测试配置系统
    print("=== 配置系统测试 ===")
    
    try:
        config = ConfigManager.load_config()
        print(f"✅ 配置加载成功")
        print(f"服务地址: {config.host}:{config.port}")
        print(f"推理引擎: {config.inference_engine}")
        print(f"推理模式: {config.inference_mode}")
        
        # 环境验证
        validation = ConfigManager.validate_environment()
        if validation["valid"]:
            print("✅ 环境验证通过")
        else:
            print("❌ 环境验证失败")
            for error in validation["errors"]:
                print(f"  - {error}")
        
        if validation["warnings"]:
            print("⚠️ 警告信息:")
            for warning in validation["warnings"]:
                print(f"  - {warning}")
                
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        import traceback
        traceback.print_exc()