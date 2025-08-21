"""
自定义异常类
定义推理服务相关的异常类型
"""


class InferenceEngineError(Exception):
    """推理引擎基础异常"""
    pass


class ModelLoadError(InferenceEngineError):
    """模型加载异常"""
    def __init__(self, model_name: str, message: str = None):
        self.model_name = model_name
        super().__init__(message or f"模型加载失败: {model_name}")


class ModelNotFoundError(InferenceEngineError):
    """模型未找到异常"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(f"模型未找到: {model_name}")


class InferenceError(InferenceEngineError):
    """推理执行异常"""
    def __init__(self, message: str, model_name: str = None):
        self.model_name = model_name
        super().__init__(f"推理失败: {message}")


class ResourceLimitError(InferenceEngineError):
    """资源限制异常"""
    def __init__(self, resource_type: str, current: float, limit: float):
        self.resource_type = resource_type
        self.current = current
        self.limit = limit
        super().__init__(f"{resource_type}资源不足: {current:.1f} > {limit:.1f}")


class ConfigurationError(InferenceEngineError):
    """配置错误异常"""
    pass


class EngineNotSupportedError(InferenceEngineError):
    """引擎不支持异常"""
    def __init__(self, engine_name: str, platform: str):
        self.engine_name = engine_name
        self.platform = platform
        super().__init__(f"引擎 {engine_name} 在平台 {platform} 上不受支持")