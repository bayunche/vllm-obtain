#!/usr/bin/env python3
"""
性能基准测试
"""

import time
import requests
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import os


class PerformanceTester:
    """性能测试器"""
    
    def __init__(self, base_url="http://127.0.0.1:8001"):
        self.base_url = base_url
        self.api_base = f"{base_url}/v1"
        self.results = []
    
    def test_single_request_latency(self, request_count=10):
        """测试单请求延迟"""
        print(f"测试单请求延迟 ({request_count}次)...")
        
        latencies = []
        payload = {
            "model": "qwen-0.5b",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 20
        }
        
        for i in range(request_count):
            start_time = time.time()
            
            try:
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    json=payload,
                    timeout=30
                )
                end_time = time.time()
                latency = end_time - start_time
                
                if response.status_code == 200:
                    latencies.append(latency)
                    print(f"  请求 {i+1}: {latency:.3f}s - 成功")
                else:
                    print(f"  请求 {i+1}: {latency:.3f}s - 失败({response.status_code})")
                    
            except Exception as e:
                end_time = time.time()
                latency = end_time - start_time
                print(f"  请求 {i+1}: {latency:.3f}s - 异常({e})")
        
        if latencies:
            avg_latency = statistics.mean(latencies)
            median_latency = statistics.median(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            print(f"延迟统计:")
            print(f"  平均: {avg_latency:.3f}s")
            print(f"  中位数: {median_latency:.3f}s") 
            print(f"  最小: {min_latency:.3f}s")
            print(f"  最大: {max_latency:.3f}s")
            print(f"  成功率: {len(latencies)}/{request_count} ({len(latencies)/request_count*100:.1f}%)")
            
            return {
                "test": "single_request_latency",
                "avg_latency": avg_latency,
                "median_latency": median_latency,
                "min_latency": min_latency,
                "max_latency": max_latency,
                "success_count": len(latencies),
                "total_count": request_count,
                "success_rate": len(latencies)/request_count*100
            }
        else:
            print("所有请求都失败了")
            return None
    
    def test_concurrent_requests(self, concurrent_users=5, requests_per_user=3):
        """测试并发请求"""
        print(f"测试并发请求 ({concurrent_users}用户 x {requests_per_user}请求)...")
        
        def single_request(request_id):
            payload = {
                "model": "qwen-0.5b",
                "messages": [{"role": "user", "content": f"Request {request_id}"}],
                "max_tokens": 15
            }
            
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    json=payload,
                    timeout=30
                )
                end_time = time.time()
                latency = end_time - start_time
                
                return {
                    "request_id": request_id,
                    "latency": latency,
                    "success": response.status_code == 200,
                    "status_code": response.status_code
                }
                
            except Exception as e:
                end_time = time.time()
                latency = end_time - start_time
                return {
                    "request_id": request_id,
                    "latency": latency,
                    "success": False,
                    "error": str(e)
                }
        
        # 执行并发请求
        total_requests = concurrent_users * requests_per_user
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [
                executor.submit(single_request, i)
                for i in range(total_requests)
            ]
            
            results = []
            for future in as_completed(futures):
                results.append(future.result())
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 分析结果
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        
        if successful_requests:
            latencies = [r['latency'] for r in successful_requests]
            avg_latency = statistics.mean(latencies)
            throughput = len(successful_requests) / total_time
        else:
            avg_latency = 0
            throughput = 0
        
        print(f"并发测试结果:")
        print(f"  总请求: {total_requests}")
        print(f"  成功请求: {len(successful_requests)}")
        print(f"  失败请求: {len(failed_requests)}")
        print(f"  成功率: {len(successful_requests)/total_requests*100:.1f}%")
        print(f"  总耗时: {total_time:.3f}s")
        print(f"  平均延迟: {avg_latency:.3f}s")
        print(f"  吞吐量: {throughput:.2f} req/s")
        
        return {
            "test": "concurrent_requests",
            "concurrent_users": concurrent_users,
            "requests_per_user": requests_per_user,
            "total_requests": total_requests,
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "success_rate": len(successful_requests)/total_requests*100,
            "total_time": total_time,
            "avg_latency": avg_latency,
            "throughput": throughput
        }
    
    def test_system_resources(self):
        """测试系统资源使用"""
        print("测试系统资源使用...")
        
        process = psutil.Process(os.getpid())
        
        # 获取系统信息
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        process_memory = process.memory_info()
        
        print(f"系统资源:")
        print(f"  CPU使用率: {cpu_percent:.1f}%")
        print(f"  系统内存: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")
        print(f"  进程内存: {process_memory.rss/1024**2:.1f}MB")
        
        return {
            "test": "system_resources",
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used/1024**3,
            "memory_total_gb": memory.total/1024**3,
            "process_memory_mb": process_memory.rss/1024**2
        }
    
    def test_health_check_speed(self, request_count=20):
        """测试健康检查速度"""
        print(f"测试健康检查速度 ({request_count}次)...")
        
        times = []
        for i in range(request_count):
            start_time = time.time()
            response = requests.get(f"{self.base_url}/health")
            end_time = time.time()
            
            response_time = end_time - start_time
            times.append(response_time)
            
            if i % 5 == 0:
                print(f"  {i+1}/{request_count}: {response_time:.4f}s")
        
        avg_time = statistics.mean(times)
        median_time = statistics.median(times)
        
        print(f"健康检查性能:")
        print(f"  平均响应时间: {avg_time:.4f}s")
        print(f"  中位数响应时间: {median_time:.4f}s")
        
        return {
            "test": "health_check_speed", 
            "avg_response_time": avg_time,
            "median_response_time": median_time,
            "request_count": request_count
        }


def main():
    """主测试函数"""
    print("=== 性能基准测试 ===")
    
    tester = PerformanceTester()
    results = []
    
    # 检查服务状态
    try:
        health_response = requests.get("http://127.0.0.1:8001/health", timeout=5)
        if health_response.status_code not in [200, 503]:
            print("服务不可用，跳过性能测试")
            return False
        
        health_data = health_response.json()
        print(f"服务状态: {health_data.get('status')}")
        print(f"已加载模型: {health_data.get('models', 0)}")
    
    except Exception as e:
        print(f"无法连接到服务: {e}")
        return False
    
    # 执行测试
    tests = [
        ("健康检查速度", lambda: tester.test_health_check_speed(20)),
        ("系统资源使用", lambda: tester.test_system_resources()),
        ("单请求延迟", lambda: tester.test_single_request_latency(5)),
        ("并发请求", lambda: tester.test_concurrent_requests(3, 2))
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"测试: {test_name}")
        print('='*40)
        
        try:
            result = test_func()
            if result:
                results.append(result)
                print(f"完成: {test_name}")
            else:
                print(f"失败: {test_name}")
                
        except Exception as e:
            print(f"异常: {test_name} - {e}")
    
    # 输出总结
    print(f"\n{'='*50}")
    print("性能测试总结")
    print('='*50)
    
    for result in results:
        test_name = result.get('test', 'unknown')
        print(f"\n{test_name}:")
        
        if 'avg_latency' in result:
            print(f"  平均延迟: {result['avg_latency']:.3f}s")
        if 'throughput' in result:
            print(f"  吞吐量: {result['throughput']:.2f} req/s")
        if 'success_rate' in result:
            print(f"  成功率: {result['success_rate']:.1f}%")
    
    print(f"\n测试完成！共执行 {len(results)} 项测试")
    return True


if __name__ == "__main__":
    main()