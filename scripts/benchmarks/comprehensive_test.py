#!/usr/bin/env python3
"""
全面测试脚本 - 生产级测试
测试并发性能、Token生成速度、系统资源使用
"""

import requests
import time
import json
import threading
import concurrent.futures
import psutil
import os
from datetime import datetime
from collections import defaultdict
import statistics

BASE_URL = "http://127.0.0.1:8001"
API_BASE = f"{BASE_URL}/v1"

class SystemMonitor:
    """系统资源监控器"""
    
    def __init__(self):
        self.cpu_samples = []
        self.memory_samples = []
        self.monitoring = False
        self.start_time = None
        
    def start_monitoring(self):
        """开始监控"""
        self.monitoring = True
        self.start_time = time.time()
        
        def monitor():
            while self.monitoring:
                cpu = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                self.cpu_samples.append({
                    'timestamp': time.time() - self.start_time,
                    'cpu_percent': cpu,
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / 1024**3,
                    'memory_available_gb': memory.available / 1024**3
                })
                
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2)
    
    def get_stats(self):
        """获取统计信息"""
        if not self.cpu_samples:
            return {}
            
        cpu_values = [s['cpu_percent'] for s in self.cpu_samples]
        memory_values = [s['memory_percent'] for s in self.cpu_samples]
        memory_used_values = [s['memory_used_gb'] for s in self.cpu_samples]
        
        return {
            'duration': time.time() - self.start_time if self.start_time else 0,
            'cpu': {
                'avg': statistics.mean(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            },
            'memory': {
                'avg_percent': statistics.mean(memory_values),
                'max_percent': max(memory_values),
                'avg_used_gb': statistics.mean(memory_used_values),
                'max_used_gb': max(memory_used_values)
            },
            'samples': len(self.cpu_samples)
        }

def single_request(request_id, prompt_length="short"):
    """单个请求测试"""
    prompts = {
        "short": f"简单回答：什么是AI？(请求{request_id})",
        "medium": f"请详细解释人工智能的三个主要应用领域，每个领域用2-3句话说明。(请求{request_id})",
        "long": f"请写一篇关于机器学习在现代社会中应用的短文，包括至少3个具体例子，每个例子都要说明其工作原理和实际效果。请确保内容准确且易于理解。(请求{request_id})"
    }
    
    max_tokens = {
        "short": 30,
        "medium": 100,
        "long": 200
    }
    
    try:
        start_time = time.time()
        
        payload = {
            "model": "qwen-0.5b",
            "messages": [
                {"role": "user", "content": prompts[prompt_length]}
            ],
            "max_tokens": max_tokens[prompt_length],
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{API_BASE}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            usage = data.get('usage', {})
            
            return {
                "request_id": request_id,
                "success": True,
                "response_time": response_time,
                "content_length": len(content),
                "prompt_tokens": usage.get('prompt_tokens', 0),
                "completion_tokens": usage.get('completion_tokens', 0),
                "total_tokens": usage.get('total_tokens', 0),
                "tokens_per_second": usage.get('completion_tokens', 0) / response_time if response_time > 0 else 0,
                "content": content[:100] + "..." if len(content) > 100 else content,
                "prompt_length": prompt_length
            }
        else:
            return {
                "request_id": request_id,
                "success": False,
                "response_time": response_time,
                "error": f"HTTP {response.status_code}: {response.text[:100]}"
            }
        
    except Exception as e:
        return {
            "request_id": request_id,
            "success": False,
            "response_time": 0,
            "error": str(e)
        }

def test_concurrent_performance(num_requests=50, max_workers=10):
    """测试并发性能"""
    print(f"\n🚀 并发性能测试")
    print(f"   请求数: {num_requests}")
    print(f"   并发数: {max_workers}")
    print("=" * 60)
    
    monitor = SystemMonitor()
    monitor.start_monitoring()
    
    start_time = time.time()
    
    # 使用不同长度的提示词测试
    prompt_types = ["short"] * (num_requests // 2) + ["medium"] * (num_requests // 3) + ["long"] * (num_requests - num_requests//2 - num_requests//3)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(single_request, i, prompt_types[i % len(prompt_types)]) for i in range(num_requests)]
        results = []
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            results.append(result)
            
            if result["success"]:
                print(f"✅ {i+1:2d}/{num_requests} - {result['response_time']:.3f}s - {result['tokens_per_second']:.1f} t/s")
            else:
                print(f"❌ {i+1:2d}/{num_requests} - 失败: {result.get('error', 'Unknown')[:50]}")
    
    total_time = time.time() - start_time
    monitor.stop_monitoring()
    system_stats = monitor.get_stats()
    
    return results, total_time, system_stats

def test_token_generation_speed():
    """测试不同长度文本的Token生成速度"""
    print(f"\n📊 Token生成速度测试")
    print("=" * 60)
    
    test_cases = [
        ("短文本", "short", 5),
        ("中等文本", "medium", 5), 
        ("长文本", "long", 5)
    ]
    
    results = {}
    
    for case_name, prompt_type, num_tests in test_cases:
        print(f"\n测试 {case_name} ({num_tests}次)...")
        case_results = []
        
        for i in range(num_tests):
            result = single_request(f"{case_name}_{i}", prompt_type)
            case_results.append(result)
            
            if result["success"]:
                print(f"  {i+1}: {result['response_time']:.3f}s - {result['completion_tokens']} tokens - {result['tokens_per_second']:.1f} t/s")
            else:
                print(f"  {i+1}: 失败 - {result.get('error', 'Unknown')}")
        
        # 统计
        successful = [r for r in case_results if r["success"]]
        if successful:
            response_times = [r["response_time"] for r in successful]
            tokens_per_sec = [r["tokens_per_second"] for r in successful]
            completion_tokens = [r["completion_tokens"] for r in successful]
            
            results[case_name] = {
                "success_rate": len(successful) / len(case_results) * 100,
                "avg_response_time": statistics.mean(response_times),
                "avg_tokens_per_second": statistics.mean(tokens_per_sec),
                "avg_completion_tokens": statistics.mean(completion_tokens),
                "max_tokens_per_second": max(tokens_per_sec),
                "min_tokens_per_second": min(tokens_per_sec)
            }
        else:
            results[case_name] = {"success_rate": 0}
    
    return results

def test_stress_limits():
    """压力极限测试"""
    print(f"\n⚡ 压力极限测试")
    print("=" * 60)
    
    # 逐步增加并发数测试
    concurrent_tests = [1, 2, 5, 10, 15, 20]
    results = {}
    
    for concurrent_num in concurrent_tests:
        print(f"\n测试并发数: {concurrent_num}")
        
        try:
            test_results, total_time, system_stats = test_concurrent_performance(
                num_requests=concurrent_num * 3,  # 每个并发数测试3倍请求
                max_workers=concurrent_num
            )
            
            successful = [r for r in test_results if r["success"]]
            success_rate = len(successful) / len(test_results) * 100
            
            if successful:
                avg_response_time = statistics.mean([r["response_time"] for r in successful])
                avg_tokens_per_sec = statistics.mean([r["tokens_per_second"] for r in successful])
                throughput = len(test_results) / total_time
                
                results[concurrent_num] = {
                    "success_rate": success_rate,
                    "avg_response_time": avg_response_time,
                    "avg_tokens_per_second": avg_tokens_per_sec,
                    "throughput": throughput,
                    "system_stats": system_stats
                }
                
                print(f"  成功率: {success_rate:.1f}%")
                print(f"  平均响应时间: {avg_response_time:.3f}s")
                print(f"  平均生成速度: {avg_tokens_per_sec:.1f} tokens/s")
                print(f"  吞吐量: {throughput:.2f} 请求/s")
                print(f"  平均CPU: {system_stats['cpu']['avg']:.1f}%")
            else:
                results[concurrent_num] = {"success_rate": 0}
                print(f"  全部失败")
                break  # 如果全部失败，停止测试更高并发
                
        except Exception as e:
            print(f"  测试失败: {e}")
            break
    
    return results

def generate_report(concurrent_results, token_speed_results, stress_results):
    """生成测试报告"""
    
    # 获取系统信息
    cpu_info = os.popen('sysctl -n machdep.cpu.brand_string').read().strip()
    cpu_cores = os.popen('sysctl -n hw.ncpu').read().strip()
    memory_total = psutil.virtual_memory().total / 1024**3
    
    report = f"""# Mac Studio 生产环境测试报告

## 测试环境

- **设备**: Mac Studio 
- **处理器**: {cpu_info}
- **核心数**: {cpu_cores}
- **内存**: {memory_total:.1f}GB
- **推理引擎**: MLX (Apple Silicon优化)
- **模型**: Qwen2.5-0.5B-Instruct
- **测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 并发性能测试结果

"""
    
    if concurrent_results:
        successful = [r for r in concurrent_results if r["success"]]
        if successful:
            response_times = [r["response_time"] for r in successful]
            tokens_per_sec_list = [r["tokens_per_second"] for r in successful]
            
            report += f"""### 基础性能指标

- **总请求数**: {len(concurrent_results)}
- **成功请求**: {len(successful)}
- **成功率**: {len(successful)/len(concurrent_results)*100:.1f}%
- **平均响应时间**: {statistics.mean(response_times):.3f}秒
- **最快响应**: {min(response_times):.3f}秒
- **最慢响应**: {max(response_times):.3f}秒
- **平均Token生成速度**: {statistics.mean(tokens_per_sec_list):.1f} tokens/秒
- **最快生成速度**: {max(tokens_per_sec_list):.1f} tokens/秒

"""

    # Token生成速度测试
    report += "## Token生成速度测试\n\n"
    
    for test_name, results in token_speed_results.items():
        if results.get("success_rate", 0) > 0:
            report += f"""### {test_name}

- **成功率**: {results['success_rate']:.1f}%
- **平均响应时间**: {results['avg_response_time']:.3f}秒
- **平均Token数**: {results['avg_completion_tokens']:.0f}
- **平均生成速度**: {results['avg_tokens_per_second']:.1f} tokens/秒
- **最快生成速度**: {results['max_tokens_per_second']:.1f} tokens/秒
- **最慢生成速度**: {results['min_tokens_per_second']:.1f} tokens/秒

"""

    # 压力测试结果
    report += "## 压力极限测试\n\n"
    report += "| 并发数 | 成功率 | 平均响应时间 | Token生成速度 | 吞吐量 | 平均CPU |\n"
    report += "|--------|--------|--------------|---------------|--------|----------|\n"
    
    for concurrent_num, result in stress_results.items():
        if result.get("success_rate", 0) > 0:
            report += f"| {concurrent_num} | {result['success_rate']:.1f}% | {result['avg_response_time']:.3f}s | {result['avg_tokens_per_second']:.1f} t/s | {result['throughput']:.2f} req/s | {result['system_stats']['cpu']['avg']:.1f}% |\n"
        else:
            report += f"| {concurrent_num} | 0% | - | - | - | - |\n"
    
    # 性能分析
    report += "\n## 性能分析\n\n"
    
    # 找到最优并发数
    best_concurrent = None
    best_throughput = 0
    
    for concurrent_num, result in stress_results.items():
        if result.get("success_rate", 0) >= 95 and result.get("throughput", 0) > best_throughput:
            best_throughput = result["throughput"]
            best_concurrent = concurrent_num
    
    if best_concurrent:
        report += f"""### 最优配置

- **推荐并发数**: {best_concurrent}
- **最大吞吐量**: {best_throughput:.2f} 请求/秒
- **在该配置下的性能**:
  - 成功率: {stress_results[best_concurrent]['success_rate']:.1f}%
  - 平均响应时间: {stress_results[best_concurrent]['avg_response_time']:.3f}秒
  - Token生成速度: {stress_results[best_concurrent]['avg_tokens_per_second']:.1f} tokens/秒

"""

    # 系统资源使用
    if stress_results:
        max_cpu_result = max(stress_results.values(), 
                           key=lambda x: x.get('system_stats', {}).get('cpu', {}).get('avg', 0))
        
        if 'system_stats' in max_cpu_result:
            stats = max_cpu_result['system_stats']
            report += f"""### 系统资源使用

- **最高CPU使用率**: {stats['cpu']['max']:.1f}%
- **平均CPU使用率**: {stats['cpu']['avg']:.1f}%
- **最高内存使用**: {stats['memory']['max_used_gb']:.1f}GB ({stats['memory']['max_percent']:.1f}%)
- **平均内存使用**: {stats['memory']['avg_used_gb']:.1f}GB ({stats['memory']['avg_percent']:.1f}%)

"""

    # 结论和建议
    report += """## 结论与建议

### 性能特点

1. **MLX引擎优势**: 在Apple Silicon上表现优秀，Token生成速度稳定
2. **内存效率**: 0.5B参数模型内存使用合理，适合单机部署
3. **并发能力**: 受限于Flask开发服务器，建议生产环境使用Gunicorn

### 生产部署建议

1. **推荐配置**:
   - 使用Gunicorn + Gevent workers
   - Worker数量: 4-8个
   - 每个Worker处理2-3个并发请求

2. **性能优化**:
   - 启用模型缓存
   - 使用SSD存储
   - 配置适当的max_tokens限制

3. **监控指标**:
   - 响应时间 < 1秒
   - CPU使用率 < 80%
   - 内存使用率 < 70%

### 适用场景

- **适合**: 中小规模应用、开发测试、原型验证
- **不适合**: 高并发生产环境（建议使用更大模型+GPU集群）

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    return report

def main():
    """主测试流程"""
    print("🎯 Mac Studio 全面性能测试")
    print("=" * 60)
    
    # 检查服务状态
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ 服务不可用")
            return
        
        data = response.json()
        if data.get('models', 0) == 0:
            print("❌ 没有加载的模型")
            return
        
        print(f"✅ 服务状态: {data.get('status')}")
        print(f"✅ 已加载模型: {data.get('models')} 个")
        
    except Exception as e:
        print(f"❌ 服务检查失败: {e}")
        return
    
    # 1. 并发性能测试
    print("\n" + "="*60)
    print("第一阶段: 并发性能测试")
    concurrent_results, total_time, system_stats = test_concurrent_performance(30, 8)
    
    # 2. Token生成速度测试  
    print("\n" + "="*60)
    print("第二阶段: Token生成速度测试")
    token_results = test_token_generation_speed()
    
    # 3. 压力极限测试
    print("\n" + "="*60)  
    print("第三阶段: 压力极限测试")
    stress_results = test_stress_limits()
    
    # 4. 生成报告
    print("\n" + "="*60)
    print("生成测试报告...")
    
    report = generate_report(concurrent_results, token_results, stress_results)
    
    # 保存报告
    report_file = f"mac测试报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ 测试完成！报告已保存到: {report_file}")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()