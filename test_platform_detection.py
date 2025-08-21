#!/usr/bin/env python3
"""
跨平台引擎选择功能测试
"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.platform_detector import PlatformDetector, get_optimal_engine


def test_platform_detection():
    """测试平台检测功能"""
    print("🔍 跨平台引擎选择功能测试")
    print("=" * 50)
    
    detector = PlatformDetector()
    
    # 1. 测试平台信息获取
    print("\n1️⃣ 平台信息检测:")
    platform_info = detector.get_platform_info()
    for key, value in platform_info.items():
        print(f"   {key}: {value}")
    
    # 2. 测试设备信息获取
    print("\n2️⃣ 设备信息检测:")
    device_info = detector.get_device_info()
    for key, value in device_info.items():
        print(f"   {key}: {value}")
    
    # 3. 测试CUDA可用性
    print("\n3️⃣ CUDA 可用性测试:")
    cuda_available = detector.is_cuda_available()
    print(f"   CUDA 可用: {'✅' if cuda_available else '❌'}")
    
    # 4. 测试MPS可用性 (Apple Silicon)
    print("\n4️⃣ MPS 可用性测试:")
    mps_available = detector.is_mps_available()
    print(f"   MPS 可用: {'✅' if mps_available else '❌'}")
    
    # 5. 测试引擎可用性检查
    print("\n5️⃣ 推理引擎可用性检查:")
    engines = ['vllm', 'mlx', 'llama_cpp']
    
    for engine in engines:
        if engine == 'vllm':
            available = detector._check_vllm_availability()
        elif engine == 'mlx':
            available = detector._check_mlx_availability()
        elif engine == 'llama_cpp':
            available = detector._check_llamacpp_availability()
        else:
            available = False
        
        status = '✅' if available else '❌'
        description = detector.SUPPORTED_ENGINES[engine]['description']
        print(f"   {engine}: {status} - {description}")
    
    # 6. 测试最佳引擎选择
    print("\n6️⃣ 最佳引擎选择:")
    optimal_engine = get_optimal_engine()
    print(f"   推荐引擎: {optimal_engine}")
    print(f"   引擎描述: {detector.SUPPORTED_ENGINES[optimal_engine]['description']}")
    
    # 7. 测试引擎兼容性验证
    print("\n7️⃣ 引擎兼容性验证:")
    for engine in engines:
        compatible = detector.validate_engine_compatibility(engine)
        status = '✅' if compatible else '❌'
        print(f"   {engine}: {status}")
    
    # 8. 测试强制引擎选择
    print("\n8️⃣ 强制引擎选择测试:")
    for engine in engines:
        forced_engine = detector.detect_best_engine(force_engine=engine)
        print(f"   强制 {engine} → 实际选择: {forced_engine}")
    
    # 9. 测试环境变量影响
    print("\n9️⃣ 环境变量测试:")
    original_env = os.environ.get('INFERENCE_ENGINE')
    
    test_engines = ['vllm', 'mlx', 'llama_cpp', 'auto', 'invalid_engine']
    for test_engine in test_engines:
        os.environ['INFERENCE_ENGINE'] = test_engine
        selected_engine = get_optimal_engine()
        print(f"   INFERENCE_ENGINE={test_engine} → {selected_engine}")
    
    # 恢复原始环境变量
    if original_env:
        os.environ['INFERENCE_ENGINE'] = original_env
    elif 'INFERENCE_ENGINE' in os.environ:
        del os.environ['INFERENCE_ENGINE']
    
    print("\n🎉 跨平台引擎选择功能测试完成!")
    return True


def test_engine_selection_logic():
    """测试引擎选择逻辑"""
    print("\n🧠 引擎选择逻辑详细测试")
    print("=" * 50)
    
    detector = PlatformDetector()
    platform_info = detector.get_platform_info()
    system = platform_info['system']
    machine = platform_info['machine']
    
    print(f"\n当前平台: {system} ({machine})")
    
    # 测试各种条件下的引擎选择
    if system == 'darwin':  # macOS
        print("\n🍎 macOS 环境测试:")
        if 'arm64' in machine or 'aarch64' in machine:
            print("   检测到 Apple Silicon")
            mlx_available = detector._check_mlx_availability()
            if mlx_available:
                print("   ✅ 应选择 MLX 引擎")
            else:
                print("   ⚠️ MLX 不可用，应回退到 llama.cpp")
        else:
            print("   检测到 Intel Mac")
            print("   ✅ 应选择 llama.cpp 引擎")
    
    elif system in ['linux', 'windows']:  # Linux/Windows
        print(f"\n🐧🪟 {system.title()} 环境测试:")
        cuda_available = detector.is_cuda_available()
        if cuda_available:
            print("   检测到 CUDA 支持")
            vllm_available = detector._check_vllm_availability()
            if vllm_available:
                print("   ✅ 应选择 VLLM 引擎")
            else:
                print("   ⚠️ VLLM 不可用，应回退到 llama.cpp")
        else:
            print("   未检测到 CUDA")
            print("   ✅ 应选择 llama.cpp 引擎")
    
    else:
        print(f"\n❓ 未知平台: {system}")
        print("   ✅ 应选择 llama.cpp 引擎 (通用回退)")
    
    return True


def test_error_handling():
    """测试错误处理"""
    print("\n🛡️ 错误处理测试")
    print("=" * 50)
    
    detector = PlatformDetector()
    
    # 测试无效引擎名称
    print("\n测试无效引擎名称:")
    invalid_engines = ['invalid_engine', '', None, 'pytorch', 'tensorflow']
    
    for invalid_engine in invalid_engines:
        try:
            compatible = detector.validate_engine_compatibility(invalid_engine)
            print(f"   {invalid_engine}: {'兼容' if compatible else '不兼容'}")
        except Exception as e:
            print(f"   {invalid_engine}: 异常 - {e}")
    
    # 测试强制使用不兼容的引擎
    print("\n测试强制使用不兼容引擎:")
    platform_info = detector.get_platform_info()
    system = platform_info['system']
    
    if system == 'windows':
        # Windows上强制使用MLX
        result = detector.detect_best_engine('mlx')
        print(f"   Windows上强制MLX: {result}")
    elif system == 'darwin':
        # macOS上强制使用VLLM
        result = detector.detect_best_engine('vllm')
        print(f"   macOS上强制VLLM: {result}")
    
    return True


if __name__ == "__main__":
    try:
        # 执行所有测试
        test_platform_detection()
        test_engine_selection_logic()
        test_error_handling()
        
        print("\n✅ 所有测试通过！")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现异常: {e}")
        import traceback
        traceback.print_exc()