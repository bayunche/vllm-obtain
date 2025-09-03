#!/usr/bin/env python3
"""
安全的并发测试脚本
测试MLX引擎的并发处理能力
"""

import asyncio
import time
import traceback
from src.engines.mlx_engine import MlxEngine
from src.core.inference_engine import EngineConfig, InferenceRequest

async def test_concurrent_inference():
    print("="*60)
    print("🧪 MLX并发推理安全测试")
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
    
    load_success = await engine.load_model(model_name, model_path)
    if not load_success:
        print(f"❌ 模型加载失败")
        return False
    print(f"✅ 模型加载成功")
    
    # 测试单个请求
    print("\n3. 测试单个请求...")
    single_request = InferenceRequest(
        model_name=model_name,
        prompt="你好",
        max_tokens=20,
        temperature=0.7
    )
    
    try:
        response = await engine.generate(single_request)
        print(f"✅ 单个请求成功: {response.text[:50]}...")
    except Exception as e:
        print(f"❌ 单个请求失败: {e}")
        return False
    
    # 测试顺序请求
    print("\n4. 测试顺序请求（3个）...")
    for i in range(3):
        request = InferenceRequest(
            model_name=model_name,
            prompt=f"测试{i+1}",
            max_tokens=10,
            temperature=0.7
        )
        try:
            response = await engine.generate(request)
            print(f"  ✅ 请求{i+1}成功")
        except Exception as e:
            print(f"  ❌ 请求{i+1}失败: {e}")
    
    # 测试并发请求（使用锁保护）
    print("\n5. 测试并发请求（带保护）...")
    
    async def safe_generate(req_id, prompt):
        """安全的推理请求"""
        try:
            request = InferenceRequest(
                model_name=model_name,
                prompt=prompt,
                max_tokens=20,
                temperature=0.7
            )
            
            start_time = time.time()
            response = await engine.generate(request)
            elapsed = time.time() - start_time
            
            return {
                'id': req_id,
                'success': True,
                'time': elapsed,
                'text': response.text[:30]
            }
        except Exception as e:
            return {
                'id': req_id,
                'success': False,
                'error': str(e)
            }
    
    # 并发执行3个请求
    prompts = ["并发测试1", "并发测试2", "并发测试3"]
    tasks = [safe_generate(i, prompt) for i, prompt in enumerate(prompts)]
    
    print("  启动3个并发请求...")
    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_time = time.time() - start_time
    
    # 分析结果
    successful = 0
    failed = 0
    for result in results:
        if isinstance(result, Exception):
            print(f"  ❌ 异常: {result}")
            failed += 1
        elif isinstance(result, dict):
            if result['success']:
                print(f"  ✅ 请求{result['id']}成功 (耗时: {result['time']:.2f}s)")
                successful += 1
            else:
                print(f"  ❌ 请求{result['id']}失败: {result.get('error', 'Unknown')}")
                failed += 1
    
    print(f"\n  总耗时: {total_time:.2f}秒")
    print(f"  成功: {successful}/{len(results)}")
    print(f"  失败: {failed}/{len(results)}")
    
    # 测试更多并发（逐步增加）
    if successful == len(results):
        print("\n6. 测试渐进式并发...")
        for concurrent_count in [2, 3, 4, 5]:
            print(f"\n  测试{concurrent_count}个并发请求...")
            
            # 添加延迟启动以避免同时冲击
            async def delayed_generate(delay, req_id, prompt):
                await asyncio.sleep(delay * 0.05)  # 50ms间隔
                return await safe_generate(req_id, prompt)
            
            prompts = [f"渐进测试{i}" for i in range(concurrent_count)]
            tasks = [delayed_generate(i, i, prompt) for i, prompt in enumerate(prompts)]
            
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                success_count = sum(1 for r in results if isinstance(r, dict) and r['success'])
                print(f"    结果: {success_count}/{concurrent_count} 成功")
                
                if success_count < concurrent_count:
                    print(f"    ⚠️ 在{concurrent_count}个并发时出现问题，停止测试")
                    break
            except Exception as e:
                print(f"    ❌ 测试失败: {e}")
                break
    
    # 卸载模型
    print("\n7. 卸载模型...")
    await engine.unload_model(model_name)
    print("✅ 模型卸载成功")
    
    return True

async def main():
    try:
        success = await test_concurrent_inference()
        if success:
            print("\n✅ 所有测试完成")
        else:
            print("\n❌ 测试未完全通过")
        return success
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)