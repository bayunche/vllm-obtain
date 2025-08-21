"""
推理引擎实现模块
包含不同推理引擎的具体实现
"""

from .llamacpp_engine import LlamaCppEngine
# from .vllm_engine import VllmEngine
# from .mlx_engine import MlxEngine

__all__ = [
    'LlamaCppEngine',
    # 'VllmEngine',
    # 'MlxEngine'
]


def create_engine(engine_type: str, config):
    """
    根据引擎类型创建推理引擎实例
    
    Args:
        engine_type: 引擎类型 (llama_cpp, vllm, mlx)
        config: 引擎配置
        
    Returns:
        推理引擎实例
    """
    if engine_type == "llama_cpp":
        return LlamaCppEngine(config)
    elif engine_type == "vllm":
        try:
            from .vllm_engine import VllmEngine
            return VllmEngine(config)
        except ImportError:
            raise ImportError("VLLM 引擎不可用，请安装 vllm 包")
    elif engine_type == "mlx":
        try:
            from .mlx_engine import MlxEngine
            return MlxEngine(config)
        except ImportError:
            raise ImportError("MLX 引擎不可用，请安装 mlx-lm 包")
    else:
        raise ValueError(f"不支持的引擎类型: {engine_type}")