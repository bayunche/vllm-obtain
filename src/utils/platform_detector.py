"""
跨平台检测工具
自动检测运行环境并选择最适合的推理引擎
"""

import platform
import os
import sys
from typing import Optional, List, Dict
from loguru import logger


class PlatformDetector:
    """平台检测器，自动选择最佳推理引擎"""
    
    SUPPORTED_ENGINES = {
        'vllm': {
            'platforms': ['linux', 'windows'],
            'requirements': ['cuda'],
            'description': 'VLLM - 高性能CUDA推理引擎'
        },
        'mlx': {
            'platforms': ['darwin'],
            'requirements': ['arm64'],
            'description': 'MLX - Apple Silicon优化引擎'
        },
        'llama_cpp': {
            'platforms': ['linux', 'windows', 'darwin'],
            'requirements': [],
            'description': 'llama.cpp - 跨平台CPU/GPU引擎'
        }
    }
    
    @staticmethod
    def get_platform_info() -> Dict[str, str]:
        """获取平台信息"""
        return {
            'system': platform.system().lower(),
            'machine': platform.machine().lower(),
            'processor': platform.processor(),
            'python_version': sys.version,
            'platform': platform.platform()
        }
    
    @staticmethod
    def is_cuda_available() -> bool:
        """检测CUDA是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    @staticmethod
    def is_mps_available() -> bool:
        """检测MPS (Metal Performance Shaders) 是否可用"""
        try:
            import torch
            return torch.backends.mps.is_available()
        except (ImportError, AttributeError):
            return False
    
    @staticmethod
    def detect_best_engine(force_engine: Optional[str] = None) -> str:
        """
        自动检测最佳推理引擎
        
        Args:
            force_engine: 强制指定引擎 (通过环境变量INFERENCE_ENGINE)
            
        Returns:
            最佳推理引擎名称
        """
        if force_engine and force_engine != 'auto':
            if force_engine in PlatformDetector.SUPPORTED_ENGINES:
                logger.info(f"强制使用推理引擎: {force_engine}")
                return force_engine
            else:
                logger.warning(f"不支持的推理引擎: {force_engine}, 将自动选择")
        
        platform_info = PlatformDetector.get_platform_info()
        system = platform_info['system']
        machine = platform_info['machine']
        
        logger.info(f"检测到平台: {system} ({machine})")
        
        # macOS 优先使用 MLX (仅限 Apple Silicon)
        if system == 'darwin':
            if 'arm64' in machine or 'aarch64' in machine:
                if PlatformDetector._check_mlx_availability():
                    logger.info("选择 MLX 引擎 (Apple Silicon 优化)")
                    return 'mlx'
                else:
                    logger.warning("MLX 不可用，回退到 llama.cpp")
                    return 'llama_cpp'
            else:
                logger.info("Intel Mac 不支持 MLX，使用 llama.cpp")
                return 'llama_cpp'
        
        # Linux/Windows 优先使用 VLLM (需要CUDA)
        elif system in ['linux', 'windows']:
            if PlatformDetector.is_cuda_available():
                if PlatformDetector._check_vllm_availability():
                    logger.info("选择 VLLM 引擎 (CUDA 优化)")
                    return 'vllm'
                else:
                    logger.warning("VLLM 不可用，回退到 llama.cpp")
                    return 'llama_cpp'
            else:
                logger.info("未检测到 CUDA，使用 llama.cpp")
                return 'llama_cpp'
        
        # 默认回退选项
        logger.info("未知平台，使用通用引擎 llama.cpp")
        return 'llama_cpp'
    
    @staticmethod
    def _check_vllm_availability() -> bool:
        """检查 VLLM 是否可用"""
        try:
            import vllm
            return True
        except ImportError:
            return False
    
    @staticmethod
    def _check_mlx_availability() -> bool:
        """检查 MLX 是否可用"""
        try:
            import mlx.core as mx
            import mlx_lm
            return True
        except ImportError:
            return False
    
    @staticmethod
    def _check_llamacpp_availability() -> bool:
        """检查 llama.cpp 是否可用"""
        try:
            import llama_cpp
            return True
        except ImportError:
            return False
    
    @staticmethod
    def get_device_info() -> Dict[str, any]:
        """获取设备信息"""
        info = {
            'cpu_count': os.cpu_count(),
            'cuda_available': PlatformDetector.is_cuda_available(),
            'mps_available': PlatformDetector.is_mps_available(),
        }
        
        # 获取GPU信息
        if info['cuda_available']:
            try:
                import torch
                info['cuda_device_count'] = torch.cuda.device_count()
                info['cuda_device_name'] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
            except:
                pass
        
        return info
    
    @staticmethod
    def validate_engine_compatibility(engine: str) -> bool:
        """验证引擎与当前平台的兼容性"""
        if engine not in PlatformDetector.SUPPORTED_ENGINES:
            return False
        
        platform_info = PlatformDetector.get_platform_info()
        system = platform_info['system']
        machine = platform_info['machine']
        
        engine_config = PlatformDetector.SUPPORTED_ENGINES[engine]
        
        # 检查平台兼容性
        if system not in engine_config['platforms']:
            return False
        
        # 检查特殊要求
        requirements = engine_config['requirements']
        if 'cuda' in requirements and not PlatformDetector.is_cuda_available():
            return False
        if 'arm64' in requirements and 'arm64' not in machine and 'aarch64' not in machine:
            return False
        
        return True


def get_optimal_engine() -> str:
    """获取最优推理引擎 (便捷函数)"""
    force_engine = os.getenv('INFERENCE_ENGINE', 'auto')
    return PlatformDetector.detect_best_engine(force_engine)


if __name__ == "__main__":
    # 测试代码
    detector = PlatformDetector()
    
    print("=== 平台信息 ===")
    platform_info = detector.get_platform_info()
    for key, value in platform_info.items():
        print(f"{key}: {value}")
    
    print("\n=== 设备信息 ===")
    device_info = detector.get_device_info()
    for key, value in device_info.items():
        print(f"{key}: {value}")
    
    print("\n=== 推理引擎选择 ===")
    optimal_engine = get_optimal_engine()
    print(f"推荐引擎: {optimal_engine}")
    
    print(f"引擎描述: {detector.SUPPORTED_ENGINES[optimal_engine]['description']}")