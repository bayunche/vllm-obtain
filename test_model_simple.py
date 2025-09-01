#!/usr/bin/env python3
"""
简单的模型加载和推理测试
"""

import asyncio
import time
from src.engines.mlx_engine import MlxEngine
from src.core.inference_engine import EngineConfig, InferenceRequest

async def main():
    print("="*60)
    print("🚀 简单模型测试")
    print("="*60)
    
    # 创建引擎配置
    config = EngineConfig(
        engine_type='mlx',
        device_type='mps',
        max_gpu_memory=0.8,
        max_cpu_threads=8,
        max_sequence_length=2048
    )
    
    # 初始化引擎
    print("\n1. 初始化MLX引擎...")
    engine = MlxEngine(config)
    init_success = await engine.initialize()
    
    if not init_success:
        print("❌ 引擎初始化失败")
        return False
    print("✅ 引擎初始化成功")
    
    # 加载模型
    print("\n2. 加载模型...")
    model_path = './models/qwen-0.5b'
    model_name = 'qwen-test'
    
    start_time = time.time()
    load_success = await engine.load_model(model_name, model_path)
    load_time = time.time() - start_time
    
    if not load_success:
        print(f"❌ 模型加载失败: {model_path}")
        return False
    print(f"✅ 模型加载成功 (耗时: {load_time:.2f}秒)")
    
    # 测试推理
    print("\n3. 测试推理功能...")
    test_prompts = [
        "你好，请介绍一下自己",
        "什么是人工智能？",
        "Python编程语言有什么特点？"
    ]
    
    all_success = True
    token_speeds = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n测试 {i}/{len(test_prompts)}: {prompt}")
        
        request = InferenceRequest(
            model_name=model_name,
            prompt=prompt,
            max_tokens=50,
            temperature=0.7
        )
        
        try:
            start_time = time.time()
            response = await engine.generate(request)
            inference_time = time.time() - start_time
            
            # 计算token速度
            total_tokens = len(response.text.split())  # 简单估算
            tokens_per_second = total_tokens / inference_time if inference_time > 0 else 0
            token_speeds.append(tokens_per_second)
            
            print(f"✅ 推理成功")
            print(f"   响应: {response.text[:100]}...")
            print(f"   耗时: {inference_time:.2f}秒")
            print(f"   速度: {tokens_per_second:.1f} tokens/秒")
            
        except Exception as e:
            print(f"❌ 推理失败: {e}")
            all_success = False
    
    # 输出总结
    print("\n" + "="*60)
    print("📊 测试总结")
    print("="*60)
    print(f"模型路径: {model_path}")
    print(f"模型加载时间: {load_time:.2f}秒")
    print(f"推理测试: {'全部通过' if all_success else '部分失败'}")
    
    if token_speeds:
        avg_speed = sum(token_speeds) / len(token_speeds)
        print(f"平均Token速度: {avg_speed:.1f} tokens/秒")
    
    # 卸载模型
    print("\n4. 卸载模型...")
    unload_success = await engine.unload_model(model_name)
    if unload_success:
        print("✅ 模型卸载成功")
    else:
        print("❌ 模型卸载失败")
    
    return all_success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)