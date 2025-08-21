#!/usr/bin/env python3
"""
并发性能测试脚本
测试Gunicorn生产服务器的并发性能
"""

import asyncio
import aiohttp
import time
import json
import psutil
import statistics
from datetime import datetime

# 测试配置
BASE_URL = "http://127.0.0.1:8001"

# 测试请求体
TEST_REQUEST = {
    "model": "qwen-0.5b",
    "messages": [
        {"role": "user", "content": "请简单介绍一下人工智能的发展历程。"}
    ],
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": False
}

class ConcurrentTester:
    def __init__(self):
        self.results = []
        self.start_time = None
        self.end_time = None
        
    async def single_request(self, session, request_id):
        """单个请求"""
        start_time = time.time()
        
        try:
            async with session.post(
                f"{BASE_URL}/v1/chat/completions",
                json=TEST_REQUEST,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    end_time = time.time()
                    
                    # 提取token信息
                    content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                    tokens = len(content.split()) if content else 0
                    response_time = end_time - start_time
                    
                    return {
                        'request_id': request_id,
                        'success': True,
                        'response_time': response_time,
                        'tokens': tokens,
                        'tokens_per_second': tokens / response_time if response_time > 0 else 0,
                        'status_code': response.status,
                        'content': content
                    }
                else:
                    end_time = time.time()
                    error_text = await response.text()
                    return {
                        'request_id': request_id,
                        'success': False,
                        'response_time': end_time - start_time,
                        'tokens': 0,
                        'tokens_per_second': 0,
                        'status_code': response.status,
                        'error': error_text[:200]
                    }
                    
        except Exception as e:
            end_time = time.time()
            return {
                'request_id': request_id,
                'success': False,
                'response_time': end_time - start_time,
                'tokens': 0,
                'tokens_per_second': 0,
                'status_code': None,
                'error': str(e)
            }

    async def run_concurrent_test(self, concurrent_count, total_requests):
        """运行并发测试"""
        print(f"\n🔄 开始并发测试: {concurrent_count}并发, {total_requests}个请求")
        
        self.results = []
        self.start_time = time.time()
        
        # 获取系统初始状态
        initial_cpu = psutil.cpu_percent(interval=1)
        initial_memory = psutil.virtual_memory()
        
        async with aiohttp.ClientSession() as session:
            # 创建信号量限制并发数
            semaphore = asyncio.Semaphore(concurrent_count)
            
            async def limited_request(request_id):
                async with semaphore:
                    return await self.single_request(session, request_id)
            
            # 创建所有任务
            tasks = [limited_request(i) for i in range(total_requests)]
            
            # 并发执行
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for result in results:
                if isinstance(result, Exception):
                    self.results.append({
                        'request_id': -1,
                        'success': False,
                        'response_time': 0,
                        'tokens': 0,
                        'tokens_per_second': 0,
                        'status_code': None,
                        'error': str(result)
                    })
                else:
                    self.results.append(result)
        
        self.end_time = time.time()
        
        # 获取系统结束状态
        final_cpu = psutil.cpu_percent(interval=1)
        final_memory = psutil.virtual_memory()
        
        # 分析结果
        return self.analyze_results(
            concurrent_count, 
            initial_cpu, final_cpu, 
            initial_memory, final_memory
        )

    def analyze_results(self, concurrent_count, initial_cpu, final_cpu, initial_memory, final_memory):
        """分析测试结果"""
        total_time = self.end_time - self.start_time
        successful_results = [r for r in self.results if r['success']]
        failed_results = [r for r in self.results if not r['success']]
        
        if not successful_results:
            return {
                'concurrent_count': concurrent_count,
                'total_requests': len(self.results),
                'successful_requests': 0,
                'failed_requests': len(failed_results),
                'success_rate': 0.0,
                'total_time': total_time,
                'requests_per_second': 0.0,
                'avg_response_time': 0.0,
                'min_response_time': 0.0,
                'max_response_time': 0.0,
                'avg_tokens_per_second': 0.0,
                'cpu_usage_change': final_cpu - initial_cpu,
                'memory_usage_change': final_memory.percent - initial_memory.percent,
                'errors': [r.get('error', 'Unknown') for r in failed_results[:5]]
            }
        
        response_times = [r['response_time'] for r in successful_results]
        tokens_per_second = [r['tokens_per_second'] for r in successful_results if r['tokens_per_second'] > 0]
        
        return {
            'concurrent_count': concurrent_count,
            'total_requests': len(self.results),
            'successful_requests': len(successful_results),
            'failed_requests': len(failed_results),
            'success_rate': len(successful_results) / len(self.results) * 100,
            'total_time': total_time,
            'requests_per_second': len(successful_results) / total_time if total_time > 0 else 0,
            'avg_response_time': statistics.mean(response_times) if response_times else 0,
            'min_response_time': min(response_times) if response_times else 0,
            'max_response_time': max(response_times) if response_times else 0,
            'p95_response_time': statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else (max(response_times) if response_times else 0),
            'avg_tokens_per_second': statistics.mean(tokens_per_second) if tokens_per_second else 0,
            'cpu_usage_change': final_cpu - initial_cpu,
            'memory_usage_change': final_memory.percent - initial_memory.percent,
            'errors': [r.get('error', 'Unknown') for r in failed_results[:3]]
        }

    def print_results(self, stats):
        """打印测试结果"""
        print(f"\n📊 测试结果 - {stats['concurrent_count']}并发:")
        print(f"   总请求数: {stats['total_requests']}")
        print(f"   成功请求: {stats['successful_requests']}")
        print(f"   失败请求: {stats['failed_requests']}")
        print(f"   成功率: {stats['success_rate']:.1f}%")
        print(f"   总耗时: {stats['total_time']:.2f}秒")
        print(f"   吞吐量: {stats['requests_per_second']:.2f} 请求/秒")
        print(f"   平均响应时间: {stats['avg_response_time']:.3f}秒")
        print(f"   最快响应: {stats['min_response_time']:.3f}秒")
        print(f"   最慢响应: {stats['max_response_time']:.3f}秒")
        print(f"   P95响应时间: {stats['p95_response_time']:.3f}秒")
        print(f"   平均Token生成速度: {stats['avg_tokens_per_second']:.1f} tokens/秒")
        print(f"   CPU使用变化: {stats['cpu_usage_change']:+.1f}%")
        print(f"   内存使用变化: {stats['memory_usage_change']:+.1f}%")
        
        if stats['errors']:
            print(f"   主要错误:")
            for i, error in enumerate(stats['errors'], 1):
                print(f"     {i}. {error}")

async def main():
    """主函数"""
    print("🚀 Gunicorn生产服务器并发性能测试")
    print("="*60)
    
    tester = ConcurrentTester()
    all_results = []
    
    # 测试不同并发级别
    test_configs = [
        (2, 10),   # 2并发, 10请求
        (5, 15),   # 5并发, 15请求  
        (8, 24),   # 8并发, 24请求
        (10, 30),  # 10并发, 30请求
        (15, 30),  # 15并发, 30请求
        (20, 40),  # 20并发, 40请求
    ]
    
    for concurrent_count, total_requests in test_configs:
        try:
            stats = await tester.run_concurrent_test(concurrent_count, total_requests)
            tester.print_results(stats)
            all_results.append(stats)
            
            # 如果成功率低于50%，停止测试
            if stats['success_rate'] < 50:
                print(f"\n⚠️ 成功率过低({stats['success_rate']:.1f}%)，停止进一步测试")
                break
                
            # 休息2秒后继续下一组测试
            await asyncio.sleep(2)
            
        except Exception as e:
            print(f"❌ 测试{concurrent_count}并发时出错: {e}")
            break
    
    # 输出汇总结果
    print("\n" + "="*60)
    print("📈 性能测试汇总")
    print("="*60)
    print(f"{'并发数':<8} {'成功率':<8} {'吞吐量':<12} {'平均响应时间':<12} {'Token速度':<12}")
    print("-"*60)
    
    for stats in all_results:
        print(f"{stats['concurrent_count']:<8} "
              f"{stats['success_rate']:<7.1f}% "
              f"{stats['requests_per_second']:<11.2f} "
              f"{stats['avg_response_time']:<11.3f} "
              f"{stats['avg_tokens_per_second']:<11.1f}")
    
    # 找出最佳性能点
    if all_results:
        best_throughput = max(all_results, key=lambda x: x['requests_per_second'])
        best_success_rate = max(all_results, key=lambda x: x['success_rate'])
        
        print(f"\n🎯 性能建议:")
        print(f"   最高吞吐量: {best_throughput['concurrent_count']}并发 "
              f"({best_throughput['requests_per_second']:.2f} 请求/秒)")
        print(f"   最高成功率: {best_success_rate['concurrent_count']}并发 "
              f"({best_success_rate['success_rate']:.1f}%)")
        
        # 推荐配置
        stable_configs = [s for s in all_results if s['success_rate'] >= 95]
        if stable_configs:
            recommended = max(stable_configs, key=lambda x: x['requests_per_second'])
            print(f"   推荐配置: {recommended['concurrent_count']}并发 "
                  f"(成功率{recommended['success_rate']:.1f}%, "
                  f"吞吐量{recommended['requests_per_second']:.2f})")

if __name__ == "__main__":
    asyncio.run(main())