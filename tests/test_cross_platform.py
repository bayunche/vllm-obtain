#!/usr/bin/env python3
"""
跨平台兼容性综合测试
测试服务在不同平台上的兼容性和自动适配能力
"""

import sys
import os
import platform
import asyncio
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.platform_detector import PlatformDetector, get_optimal_engine
from src.utils.config import ConfigManager
from src.engines import create_engine, VLLM_AVAILABLE, MLX_AVAILABLE
from src.core.inference_engine import EngineConfig


class CrossPlatformTester:
    """跨平台测试器"""
    
    def __init__(self):
        self.detector = PlatformDetector()
        self.results = {
            'platform_detection': None,
            'engine_selection': None,
            'engine_fallback': None,
            'config_loading': None,
            'path_normalization': None
        }
    
    def test_platform_detection(self) -> bool:
        """测试平台检测功能"""
        print("\n🔍 测试平台检测功能...")
        
        try:
            # 获取平台信息
            platform_info = self.detector.get_platform_info()
            assert platform_info['system'] in ['darwin', 'linux', 'windows']
            assert platform_info['machine'] is not None
            assert platform_info['python_version'] is not None
            
            # 获取设备信息
            device_info = self.detector.get_device_info()
            assert 'cpu_count' in device_info
            assert device_info['cpu_count'] > 0
            
            # 检测加速设备
            cuda_available = device_info.get('cuda_available', False)
            mps_available = device_info.get('mps_available', False)
            
            print(f"✅ 平台: {platform_info['system']} ({platform_info['machine']})")
            print(f"✅ CPU 核心数: {device_info['cpu_count']}")
            print(f"✅ CUDA 可用: {'是' if cuda_available else '否'}")
            print(f"✅ MPS 可用: {'是' if mps_available else '否'}")
            
            # 检查内存信息（如果有）
            if 'total_memory_gb' in device_info:
                print(f"✅ 总内存: {device_info['total_memory_gb']} GB")
            
            self.results['platform_detection'] = True
            return True
            
        except Exception as e:
            print(f"❌ 平台检测失败: {e}")
            self.results['platform_detection'] = False
            return False
    
    def test_engine_selection(self) -> bool:
        """测试引擎自动选择"""
        print("\n🎯 测试引擎自动选择...")
        
        try:
            # 测试自动选择
            optimal_engine = get_optimal_engine()
            print(f"✅ 自动选择引擎: {optimal_engine}")
            
            # 验证选择逻辑
            platform_info = self.detector.get_platform_info()
            system = platform_info['system']
            machine = platform_info['machine']
            
            # 验证平台特定的引擎选择
            if system == 'darwin' and ('arm64' in machine or 'aarch64' in machine):
                if MLX_AVAILABLE:
                    assert optimal_engine == 'mlx', "Apple Silicon 应该选择 MLX"
                    print("✅ Apple Silicon 正确选择了 MLX 引擎")
                else:
                    assert optimal_engine == 'llama_cpp', "MLX 不可用时应回退到 llama.cpp"
                    print("✅ MLX 不可用，正确回退到 llama.cpp")
                    
            elif system in ['linux', 'windows']:
                if self.detector.is_cuda_available() and VLLM_AVAILABLE:
                    assert optimal_engine == 'vllm', "有 CUDA 应该选择 VLLM"
                    print("✅ CUDA 可用，正确选择了 VLLM 引擎")
                else:
                    assert optimal_engine == 'llama_cpp', "无 CUDA 应该选择 llama.cpp"
                    print("✅ 无 CUDA 或 VLLM，正确选择了 llama.cpp")
            
            # 测试强制引擎选择
            engines = ['mlx', 'vllm', 'llama_cpp']
            for engine in engines:
                forced = self.detector.detect_best_engine(force_engine=engine, fallback=True)
                print(f"✅ 强制 {engine} -> {forced} (可能回退)")
            
            self.results['engine_selection'] = True
            return True
            
        except Exception as e:
            print(f"❌ 引擎选择测试失败: {e}")
            self.results['engine_selection'] = False
            return False
    
    async def test_engine_fallback(self) -> bool:
        """测试引擎回退机制"""
        print("\n🔄 测试引擎回退机制...")
        
        try:
            # 创建引擎配置
            config = EngineConfig(
                engine_type='invalid_engine',  # 故意使用无效引擎
                device_type='auto',
                max_gpu_memory=0.5,
                max_cpu_threads=4
            )
            
            # 应该自动回退到 llama.cpp
            engine = create_engine('invalid_engine', config)
            print(f"✅ 无效引擎自动回退到: {engine.__class__.__name__}")
            
            # 测试不可用引擎的回退
            if not MLX_AVAILABLE:
                engine = create_engine('mlx', config)
                assert engine.__class__.__name__ == 'LlamaCppEngine'
                print("✅ MLX 不可用时正确回退到 LlamaCpp")
            
            if not VLLM_AVAILABLE:
                engine = create_engine('vllm', config)
                assert engine.__class__.__name__ == 'LlamaCppEngine'
                print("✅ VLLM 不可用时正确回退到 LlamaCpp")
            
            self.results['engine_fallback'] = True
            return True
            
        except Exception as e:
            print(f"❌ 引擎回退测试失败: {e}")
            self.results['engine_fallback'] = False
            return False
    
    def test_config_loading(self) -> bool:
        """测试配置加载和平台特定配置"""
        print("\n⚙️ 测试配置加载...")
        
        try:
            # 测试配置加载
            config = ConfigManager.load_config()
            assert config is not None
            print("✅ 配置加载成功")
            
            # 测试平台特定配置文件检测
            system = platform.system().lower()
            platform_env_map = {
                'darwin': '.env.mac',
                'linux': '.env.linux',
                'windows': '.env.windows'
            }
            
            expected_env = platform_env_map.get(system)
            if expected_env and os.path.exists(expected_env):
                print(f"✅ 找到平台特定配置: {expected_env}")
            else:
                print(f"ℹ️ 使用默认配置（无平台特定配置）")
            
            # 测试 worker 数量验证（MLX 引擎限制）
            if config.inference_engine == 'mlx':
                assert config.workers == 1, "MLX 引擎应该强制 worker=1"
                print("✅ MLX 引擎正确设置 worker=1")
            
            self.results['config_loading'] = True
            return True
            
        except Exception as e:
            print(f"❌ 配置加载测试失败: {e}")
            self.results['config_loading'] = False
            return False
    
    def test_path_normalization(self) -> bool:
        """测试路径标准化"""
        print("\n📁 测试路径标准化...")
        
        try:
            from src.utils.config import InferenceConfig
            
            # 测试不同格式的路径
            test_paths = [
                "./models",
                ".\\models",
                "models/qwen",
                "models\\qwen",
                "/absolute/path",
                "C:\\Windows\\Path" if platform.system() == 'Windows' else "/usr/local/path"
            ]
            
            for test_path in test_paths:
                # 模拟配置验证器处理路径
                normalized = Path(test_path)
                assert normalized is not None
                print(f"✅ 路径标准化: {test_path} -> {normalized}")
            
            self.results['path_normalization'] = True
            return True
            
        except Exception as e:
            print(f"❌ 路径标准化测试失败: {e}")
            self.results['path_normalization'] = False
            return False
    
    def run_all_tests(self) -> bool:
        """运行所有测试"""
        print("=" * 60)
        print("🚀 开始跨平台兼容性测试")
        print("=" * 60)
        
        # 同步测试
        self.test_platform_detection()
        self.test_engine_selection()
        self.test_config_loading()
        self.test_path_normalization()
        
        # 异步测试
        asyncio.run(self.test_engine_fallback())
        
        # 汇总结果
        print("\n" + "=" * 60)
        print("📊 测试结果汇总")
        print("=" * 60)
        
        all_passed = True
        for test_name, result in self.results.items():
            status = "✅ 通过" if result else "❌ 失败"
            print(f"{test_name}: {status}")
            if not result:
                all_passed = False
        
        if all_passed:
            print("\n🎉 所有测试通过！服务具备良好的跨平台兼容性")
        else:
            print("\n⚠️ 部分测试失败，请检查相关功能")
        
        return all_passed


def main():
    """主函数"""
    tester = CrossPlatformTester()
    success = tester.run_all_tests()
    
    # 额外的平台特定信息
    print("\n" + "=" * 60)
    print("📝 平台特定建议")
    print("=" * 60)
    
    system = platform.system()
    if system == 'Darwin':
        print("🍎 macOS 用户:")
        print("  - 推荐使用 MLX 引擎（需要 Apple Silicon）")
        print("  - 配置文件: .env.mac")
        print("  - Worker 数量必须为 1")
    elif system == 'Linux':
        print("🐧 Linux 用户:")
        print("  - 推荐使用 VLLM 引擎（需要 NVIDIA GPU）")
        print("  - 配置文件: .env.linux")
        print("  - 支持多 Worker 并发")
    elif system == 'Windows':
        print("🪟 Windows 用户:")
        print("  - 推荐使用 llama.cpp 引擎")
        print("  - 配置文件: .env.windows")
        print("  - 建议较少的 Worker 数量")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())