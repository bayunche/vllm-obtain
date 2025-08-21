#!/usr/bin/env python3
"""
压力测试脚本
"""

import requests
import time
import json
import concurrent.futures
from datetime import datetime
import statistics

BASE_URL = "http://127.0.0.1:8001"
API_BASE = f"{BASE_URL}/v1"

def single_request(request_id):
    """单个请求测试"""
    try:
        start_time = time.time()
        
        payload = {
            "model": "qwen-0.5b",
            "messages": [
                {"role": "user", "content": f"测试请求 {request_id}: 你好"}
            ],
            "max_tokens": 50,
            "temperature": 0.7  # 修正参数名
        }
        
        response = requests.post(
            f"{API_BASE}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        response_time = time.time() - start_time
        
        return {
            "request_id": request_id,
            "status_code": response.status_code,
            "response_time": response_time,
            "success": response.status_code == 200,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "request_id": request_id,
            "status_code": 0,
            "response_time": 0,
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def run_concurrent_test(num_requests=10, num_workers=5):
    """并发测试"""
    print(f"\n🚀 开始并发压力测试")
    print(f"   总请求数: {num_requests}")
    print(f"   并发线程: {num_workers}")
    print("=" * 50)
    
    results = []
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(single_request, i) for i in range(num_requests)]
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            
            # 实时显示进度
            if result["success"]:
                print(f"✅ 请求 {result['request_id']}: {result['response_time']:.3f}s")
            else:
                print(f"❌ 请求 {result['request_id']}: 失败")
    
    total_time = time.time() - start_time
    
    # 统计分析
    successful_requests = [r for r in results if r["success"]]
    failed_requests = [r for r in results if not r["success"]]
    
    if successful_requests:
        response_times = [r["response_time"] for r in successful_requests]
        avg_response_time = statistics.mean(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        median_response_time = statistics.median(response_times)
        
        if len(response_times) > 1:
            stdev_response_time = statistics.stdev(response_times)
        else:
            stdev_response_time = 0
    else:
        avg_response_time = 0
        min_response_time = 0
        max_response_time = 0
        median_response_time = 0
        stdev_response_time = 0
    
    print("\n" + "=" * 50)
    print("📊 压力测试结果")
    print("=" * 50)
    print(f"✅ 成功请求: {len(successful_requests)}/{num_requests}")
    print(f"❌ 失败请求: {len(failed_requests)}/{num_requests}")
    print(f"📈 成功率: {len(successful_requests)/num_requests*100:.1f}%")
    print(f"⏱️ 总耗时: {total_time:.2f}秒")
    print(f"🚀 吞吐量: {num_requests/total_time:.2f} 请求/秒")
    
    if successful_requests:
        print(f"\n📊 响应时间统计:")
        print(f"   平均值: {avg_response_time:.3f}秒")
        print(f"   中位数: {median_response_time:.3f}秒")
        print(f"   最小值: {min_response_time:.3f}秒")
        print(f"   最大值: {max_response_time:.3f}秒")
        print(f"   标准差: {stdev_response_time:.3f}秒")
    
    return results

def run_load_test():
    """渐进式负载测试"""
    print("\n🔬 渐进式负载测试")
    print("=" * 50)
    
    test_configs = [
        (10, 1, "低负载测试"),
        (20, 5, "中负载测试"),
        (50, 10, "高负载测试"),
    ]
    
    all_results = []
    
    for num_requests, num_workers, test_name in test_configs:
        print(f"\n📍 {test_name}")
        results = run_concurrent_test(num_requests, num_workers)
        all_results.append({
            "test_name": test_name,
            "config": {"requests": num_requests, "workers": num_workers},
            "results": results
        })
        
        # 测试间隔
        print("\n⏸️ 等待5秒后继续下一轮测试...")
        time.sleep(5)
    
    return all_results

def check_service_health():
    """检查服务健康状态"""
    print("🏥 检查服务状态...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 服务状态: {data.get('status', 'unknown')}")
            print(f"📚 已加载模型数: {data.get('models', 0)}")
            return True
        else:
            print(f"❌ 服务响应异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 无法连接到服务: {e}")
        return False

def main():
    print("🎯 VLLM 推理服务压力测试")
    print("=" * 50)
    
    # 检查服务状态
    if not check_service_health():
        print("⚠️ 服务不可用，退出测试")
        return
    
    # 运行测试
    print("\n选择测试模式:")
    print("1. 快速测试 (10个请求)")
    print("2. 标准压力测试 (50个请求)")
    print("3. 渐进式负载测试")
    
    choice = input("\n请选择 (1-3): ").strip()
    
    if choice == "1":
        run_concurrent_test(10, 3)
    elif choice == "2":
        run_concurrent_test(50, 10)
    elif choice == "3":
        run_load_test()
    else:
        print("无效选择，运行默认快速测试")
        run_concurrent_test(10, 3)
    
    print("\n✨ 测试完成!")

if __name__ == "__main__":
    main()