#!/usr/bin/env python3
"""
串行压力测试脚本 - 避免并发问题
"""

import requests
import time
import json
from datetime import datetime

BASE_URL = "http://127.0.0.1:8001"
API_BASE = f"{BASE_URL}/v1"

def single_request(request_id):
    """单个请求测试"""
    try:
        start_time = time.time()
        
        payload = {
            "model": "qwen-0.5b",
            "messages": [
                {"role": "user", "content": f"请简单回答：什么是AI? (请求{request_id})"}
            ],
            "max_tokens": 30,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{API_BASE}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            tokens = data.get('usage', {}).get('total_tokens', 0)
            
            return {
                "request_id": request_id,
                "status_code": response.status_code,
                "response_time": response_time,
                "success": True,
                "content": content[:100] + "..." if len(content) > 100 else content,
                "tokens": tokens,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "request_id": request_id,
                "status_code": response.status_code,
                "response_time": response_time,
                "success": False,
                "error": response.text,
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

def run_serial_test(num_requests=20):
    """串行压力测试"""
    print(f"\n🚀 开始串行压力测试")
    print(f"   总请求数: {num_requests}")
    print(f"   执行模式: 串行")
    print("=" * 50)
    
    results = []
    start_time = time.time()
    
    for i in range(num_requests):
        print(f"正在执行请求 {i+1}/{num_requests}...", end=" ")
        
        result = single_request(i)
        results.append(result)
        
        if result["success"]:
            print(f"✅ 成功 ({result['response_time']:.2f}s, {result['tokens']} tokens)")
        else:
            print(f"❌ 失败 - {result.get('error', 'Unknown error')[:50]}")
        
        # 短暂停顿避免过载
        time.sleep(0.1)
    
    total_time = time.time() - start_time
    
    # 统计分析
    successful_requests = [r for r in results if r["success"]]
    failed_requests = [r for r in results if not r["success"]]
    
    print("\n" + "=" * 50)
    print("📊 压力测试结果")
    print("=" * 50)
    print(f"✅ 成功请求: {len(successful_requests)}/{num_requests}")
    print(f"❌ 失败请求: {len(failed_requests)}/{num_requests}")
    print(f"📈 成功率: {len(successful_requests)/num_requests*100:.1f}%")
    print(f"⏱️ 总耗时: {total_time:.2f}秒")
    print(f"🚀 平均吞吐量: {num_requests/total_time:.2f} 请求/秒")
    
    if successful_requests:
        response_times = [r["response_time"] for r in successful_requests]
        tokens_list = [r.get("tokens", 0) for r in successful_requests]
        
        avg_response_time = sum(response_times) / len(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        avg_tokens = sum(tokens_list) / len(tokens_list)
        
        print(f"\n📊 性能指标:")
        print(f"   平均响应时间: {avg_response_time:.3f}秒")
        print(f"   最快响应: {min_response_time:.3f}秒")
        print(f"   最慢响应: {max_response_time:.3f}秒")
        print(f"   平均Token数: {avg_tokens:.1f}")
        
        if avg_response_time > 0:
            avg_tokens_per_sec = avg_tokens / avg_response_time
            print(f"   平均生成速度: {avg_tokens_per_sec:.1f} tokens/秒")
    
    if failed_requests:
        print(f"\n❌ 失败原因统计:")
        error_counts = {}
        for req in failed_requests:
            error = req.get('error', 'Unknown')[:50]
            error_counts[error] = error_counts.get(error, 0) + 1
        
        for error, count in error_counts.items():
            print(f"   {error}: {count}次")
    
    return results

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
    print("🎯 VLLM 串行压力测试")
    print("=" * 50)
    
    # 检查服务状态
    if not check_service_health():
        print("⚠️ 服务不可用，退出测试")
        return
    
    # 运行测试
    print("\n选择测试规模:")
    print("1. 轻量测试 (10个请求)")
    print("2. 标准测试 (20个请求)")
    print("3. 重度测试 (50个请求)")
    
    choice = input("\n请选择 (1-3): ").strip()
    
    if choice == "1":
        run_serial_test(10)
    elif choice == "2":
        run_serial_test(20)
    elif choice == "3":
        run_serial_test(50)
    else:
        print("无效选择，运行默认标准测试")
        run_serial_test(20)
    
    print("\n✨ 测试完成!")

if __name__ == "__main__":
    main()