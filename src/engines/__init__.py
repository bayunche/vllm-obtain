"""
推理引擎实现模块
包含不同推理引擎的具体实现
"""

# 条件导入，避免在不支持的平台上导入失败
from .llamacpp_engine import LlamaCppEngine

# 尝试导入 VLLM 引擎（Linux/Windows）
try:
    from .vllm_engine import VllmEngine
    VLLM_AVAILABLE = True
except ImportError:
    VllmEngine = None
    VLLM_AVAILABLE = False

# 尝试导入 MLX 引擎（macOS）
try:
    from .mlx_engine import MlxEngine
    MLX_AVAILABLE = True
except ImportError:
    MlxEngine = None
    MLX_AVAILABLE = False

__all__ = [
    'LlamaCppEngine',
    'VllmEngine',
    'MlxEngine',
    'VLLM_AVAILABLE',
    'MLX_AVAILABLE'
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
    if engine_type == "llama_cpp" or engine_type == "llamacpp":
        return LlamaCppEngine(config)
    elif engine_type == "vllm":
        if not VLLM_AVAILABLE or VllmEngine is None:
            # 自动回退到 llama.cpp
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("VLLM 引擎不可用，自动回退到 llama.cpp 引擎")
            return LlamaCppEngine(config)
        return VllmEngine(config)
    elif engine_type == "mlx":
        if not MLX_AVAILABLE or MlxEngine is None:
            # 自动回退到 llama.cpp
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("MLX 引擎不可用，自动回退到 llama.cpp 引擎")
            return LlamaCppEngine(config)
        return MlxEngine(config)
    else:
        # 未知引擎类型，默认使用 llama.cpp
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"不支持的引擎类型: {engine_type}，使用默认的 llama.cpp 引擎")
        return LlamaCppEngine(config)