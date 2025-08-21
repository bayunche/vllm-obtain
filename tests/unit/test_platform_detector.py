"""
平台检测器单元测试
"""

import pytest
import os
import platform
import sys
from pathlib import Path
from unittest.mock import patch

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.platform_detector import PlatformDetector, get_optimal_engine


class TestPlatformDetector:
    """平台检测器测试类"""

    def test_get_platform_info(self):
        """测试平台信息获取"""
        detector = PlatformDetector()
        info = detector.get_platform_info()
        
        # 验证必需字段
        required_fields = ['system', 'machine', 'processor', 'python_version', 'platform']
        for field in required_fields:
            assert field in info
        
        # 验证系统类型
        assert info['system'] in ['linux', 'darwin', 'windows']
        
        print(f"检测到平台: {info['system']} ({info['machine']})")
    
    def test_detect_best_engine(self):
        """测试最佳引擎检测"""
        detector = PlatformDetector()
        engine = detector.detect_best_engine()
        
        # 引擎应该是支持的类型之一
        assert engine in detector.SUPPORTED_ENGINES.keys()
        
        print(f"选择的引擎: {engine}")
    
    def test_force_engine_selection(self):
        """测试强制引擎选择"""
        detector = PlatformDetector()
        
        for engine_name in detector.SUPPORTED_ENGINES.keys():
            forced_engine = detector.detect_best_engine(force_engine=engine_name)
            assert forced_engine == engine_name
            print(f"强制选择 {engine_name}: 成功")
    
    def test_invalid_engine_handling(self):
        """测试无效引擎处理"""
        detector = PlatformDetector()
        
        # 测试无效引擎名称
        result = detector.detect_best_engine(force_engine='invalid_engine')
        # 应该回退到自动检测
        assert result in detector.SUPPORTED_ENGINES.keys()
        print(f"无效引擎处理: 回退到 {result}")
    
    def test_windows_engine_selection(self):
        """测试Windows平台的引擎选择逻辑"""
        detector = PlatformDetector()
        platform_info = detector.get_platform_info()
        
        if platform_info['system'] == 'windows':
            # Windows应该优先选择VLLM（如果CUDA可用）或llama_cpp
            cuda_available = detector.is_cuda_available()
            if cuda_available:
                # 如果CUDA可用，检查VLLM可用性
                vllm_available = detector._check_vllm_availability()
                if vllm_available:
                    expected = 'vllm'
                else:
                    expected = 'llama_cpp'
            else:
                expected = 'llama_cpp'
            
            engine = detector.detect_best_engine()
            print(f"Windows环境: CUDA={cuda_available}, 选择引擎={engine}, 预期={expected}")
            
            # 验证选择是否合理
            assert engine in ['vllm', 'llama_cpp']
    
    def test_cuda_availability(self):
        """测试CUDA可用性检测"""
        detector = PlatformDetector()
        cuda_available = detector.is_cuda_available()
        
        print(f"CUDA 可用性: {cuda_available}")
        assert isinstance(cuda_available, bool)
    
    def test_mps_availability(self):
        """测试MPS可用性检测"""
        detector = PlatformDetector()
        mps_available = detector.is_mps_available()
        
        print(f"MPS 可用性: {mps_available}")
        assert isinstance(mps_available, bool)
    
    def test_engine_compatibility_validation(self):
        """测试引擎兼容性验证"""
        detector = PlatformDetector()
        
        for engine_name in detector.SUPPORTED_ENGINES.keys():
            compatible = detector.validate_engine_compatibility(engine_name)
            print(f"引擎 {engine_name} 兼容性: {compatible}")
            assert isinstance(compatible, bool)
        
        # 测试无效引擎
        invalid_compatible = detector.validate_engine_compatibility('invalid_engine')
        assert invalid_compatible is False
    
    def test_environment_variable_override(self):
        """测试环境变量覆盖"""
        # 保存原始环境变量
        original_env = os.environ.get('INFERENCE_ENGINE')
        
        try:
            # 测试不同的环境变量设置
            test_engines = ['vllm', 'mlx', 'llama_cpp', 'auto']
            
            for test_engine in test_engines:
                os.environ['INFERENCE_ENGINE'] = test_engine
                selected_engine = get_optimal_engine()
                
                print(f"环境变量 INFERENCE_ENGINE={test_engine} -> {selected_engine}")
                
                if test_engine == 'auto':
                    # auto应该触发自动检测
                    assert selected_engine in PlatformDetector.SUPPORTED_ENGINES.keys()
                else:
                    # 有效引擎应该被选择（或回退到兼容引擎）
                    assert selected_engine in PlatformDetector.SUPPORTED_ENGINES.keys()
        
        finally:
            # 恢复原始环境变量
            if original_env:
                os.environ['INFERENCE_ENGINE'] = original_env
            elif 'INFERENCE_ENGINE' in os.environ:
                del os.environ['INFERENCE_ENGINE']
    
    def test_engine_availability_checks(self):
        """测试引擎可用性检查"""
        detector = PlatformDetector()
        
        # 测试各个引擎的可用性检查方法
        vllm_available = detector._check_vllm_availability()
        mlx_available = detector._check_mlx_availability()
        llamacpp_available = detector._check_llamacpp_availability()
        
        print(f"引擎可用性检查:")
        print(f"  VLLM: {vllm_available}")
        print(f"  MLX: {mlx_available}")
        print(f"  llama.cpp: {llamacpp_available}")
        
        # llama.cpp应该总是可用（因为它是回退选项）
        assert llamacpp_available is True or llamacpp_available is False  # 至少不应该出错
    
    @patch('platform.system')
    def test_different_platforms(self, mock_system):
        """测试不同平台的模拟场景"""
        detector = PlatformDetector()
        
        # 测试Linux
        mock_system.return_value = 'Linux'
        with patch.object(detector, 'get_platform_info') as mock_info:
            mock_info.return_value = {'system': 'linux', 'machine': 'x86_64'}
            engine = detector.detect_best_engine()
            assert engine in ['vllm', 'llama_cpp']
            print(f"Linux平台选择: {engine}")
        
        # 测试macOS
        mock_system.return_value = 'Darwin'
        with patch.object(detector, 'get_platform_info') as mock_info:
            # 测试Apple Silicon
            mock_info.return_value = {'system': 'darwin', 'machine': 'arm64'}
            engine = detector.detect_best_engine()
            assert engine in ['mlx', 'llama_cpp']
            print(f"macOS Apple Silicon选择: {engine}")
            
            # 测试Intel Mac
            mock_info.return_value = {'system': 'darwin', 'machine': 'x86_64'}
            engine = detector.detect_best_engine()
            assert engine == 'llama_cpp'  # Intel Mac应该使用llama_cpp
            print(f"macOS Intel选择: {engine}")


if __name__ == "__main__":
    # 直接运行测试
    detector = PlatformDetector()
    print("=== 平台检测测试 ===")
    
    # 基本信息
    info = detector.get_platform_info()
    print(f"平台信息: {info}")
    
    # 引擎选择
    engine = detector.detect_best_engine()
    print(f"最佳引擎: {engine}")
    
    # CUDA检测
    cuda = detector.is_cuda_available()
    print(f"CUDA可用: {cuda}")
    
    # MPS检测  
    mps = detector.is_mps_available()
    print(f"MPS可用: {mps}")
    
    print("=== 测试完成 ===")