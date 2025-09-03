#!/usr/bin/env python3
"""
跨平台兼容性测试报告生成器
自动运行测试并生成HTML格式的测试报告
"""

import sys
import os
import platform
import json
import time
import traceback
import asyncio
import statistics
import threading
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.platform_detector import PlatformDetector, get_optimal_engine
from src.utils.config import ConfigManager
from src.engines import VLLM_AVAILABLE, MLX_AVAILABLE, create_engine
from src.core.inference_engine import EngineConfig, InferenceRequest


class CompatibilityTestReporter:
    """兼容性测试报告生成器"""
    
    def __init__(self):
        self.detector = PlatformDetector()
        self.test_results = []
        self.performance_results = {}
        self.engine_instance = None
        self.summary = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'test_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'platform': platform.platform(),
            'python_version': platform.python_version()
        }
        
    def run_test(self, test_name: str, test_func, category: str = "功能测试") -> Dict:
        """运行单个测试并记录结果"""
        result = {
            'name': test_name,
            'category': category,
            'status': 'pending',
            'message': '',
            'details': [],
            'duration': 0
        }
        
        start_time = time.time()
        try:
            test_result = test_func()
            result['duration'] = round(time.time() - start_time, 3)
            
            if isinstance(test_result, tuple):
                success, message, details = test_result
                result['status'] = 'passed' if success else 'failed'
                result['message'] = message
                result['details'] = details if isinstance(details, list) else [details]
            else:
                result['status'] = 'passed' if test_result else 'failed'
                result['message'] = '测试通过' if test_result else '测试失败'
                
        except Exception as e:
            result['status'] = 'failed'
            result['message'] = str(e)
            result['details'] = [traceback.format_exc()]
            result['duration'] = round(time.time() - start_time, 3)
            
        self.test_results.append(result)
        self.summary['total_tests'] += 1
        if result['status'] == 'passed':
            self.summary['passed'] += 1
        elif result['status'] == 'failed':
            self.summary['failed'] += 1
        else:
            self.summary['warnings'] += 1
            
        return result
    
    def test_platform_detection(self):
        """测试平台检测"""
        try:
            platform_info = self.detector.get_platform_info()
            device_info = self.detector.get_device_info()
            
            details = []
            details.append(f"操作系统: {platform_info['system']}")
            details.append(f"架构: {platform_info['machine']}")
            details.append(f"Python版本: {platform_info['python_version']}")
            details.append(f"CPU核心数: {device_info.get('cpu_count', '未知')}")
            
            if 'total_memory_gb' in device_info:
                details.append(f"总内存: {device_info['total_memory_gb']} GB")
            
            if device_info.get('cuda_available'):
                details.append(f"CUDA: 可用")
                if 'cuda_devices' in device_info:
                    for device in device_info['cuda_devices']:
                        details.append(f"  - {device['name']} ({device['memory_gb']} GB)")
            else:
                details.append(f"CUDA: 不可用")
                
            if device_info.get('mps_available'):
                details.append(f"Metal (MPS): 可用")
            else:
                details.append(f"Metal (MPS): 不可用")
                
            return True, "平台检测成功", details
            
        except Exception as e:
            return False, f"平台检测失败: {str(e)}", []
    
    def test_engine_availability(self):
        """测试引擎可用性"""
        details = []
        engines_status = {
            'MLX': ('可用' if MLX_AVAILABLE else '不可用', MLX_AVAILABLE),
            'VLLM': ('可用' if VLLM_AVAILABLE else '不可用', VLLM_AVAILABLE),
            'LlamaCpp': ('可用', True)  # 总是可用作为后备
        }
        
        all_ok = True
        for engine, (status, available) in engines_status.items():
            details.append(f"{engine}: {status}")
            if engine != 'LlamaCpp' and not available:
                all_ok = False
                
        # 检查平台推荐引擎
        optimal_engine = get_optimal_engine()
        details.append(f"推荐引擎: {optimal_engine}")
        
        message = "所有引擎状态正常" if all_ok else "部分引擎不可用（将自动回退）"
        return True, message, details
    
    def test_config_validation(self):
        """测试配置系统"""
        try:
            details = []
            
            # 测试配置加载
            config = ConfigManager.load_config()
            details.append(f"配置加载: 成功")
            
            # 检查平台特定配置
            system = platform.system().lower()
            platform_env_map = {
                'darwin': '.env.mac',
                'linux': '.env.linux',
                'windows': '.env.windows'
            }
            
            expected_env = platform_env_map.get(system)
            if expected_env and os.path.exists(expected_env):
                details.append(f"平台配置文件: {expected_env} (存在)")
            else:
                details.append(f"平台配置文件: 使用默认配置")
            
            # 验证关键配置
            details.append(f"推理引擎: {config.inference_engine}")
            details.append(f"设备类型: {config.device_type}")
            details.append(f"Worker数: {config.workers}")
            details.append(f"端口: {config.port}")
            
            # 验证环境
            validation = ConfigManager.validate_environment()
            if validation['valid']:
                details.append("环境验证: 通过")
            else:
                details.append("环境验证: 失败")
                for error in validation.get('errors', []):
                    details.append(f"  错误: {error}")
                    
            for warning in validation.get('warnings', []):
                details.append(f"  警告: {warning}")
                
            return validation['valid'], "配置系统正常" if validation['valid'] else "配置存在问题", details
            
        except Exception as e:
            return False, f"配置测试失败: {str(e)}", []
    
    def test_path_compatibility(self):
        """测试路径兼容性"""
        details = []
        test_paths = {
            '相对路径': './models',
            '反斜杠路径': '.\\cache',
            '混合路径': 'models/qwen\\model.bin'
        }
        
        all_ok = True
        for desc, test_path in test_paths.items():
            try:
                normalized = Path(test_path)
                details.append(f"{desc}: {test_path} → {normalized} ✓")
            except Exception as e:
                details.append(f"{desc}: {test_path} → 失败 ({e})")
                all_ok = False
                
        return all_ok, "路径处理正常" if all_ok else "路径处理存在问题", details
    
    def test_engine_fallback(self):
        """测试引擎回退机制"""
        try:
            from src.engines import create_engine
            from src.core.inference_engine import EngineConfig
            
            details = []
            
            # 测试无效引擎回退
            config = EngineConfig(
                engine_type='invalid_engine',
                device_type='auto'
            )
            
            engine = create_engine('invalid_engine', config)
            details.append(f"无效引擎 → {engine.__class__.__name__} (自动回退)")
            
            # 测试不可用引擎回退
            if not MLX_AVAILABLE:
                engine = create_engine('mlx', config)
                details.append(f"MLX不可用 → {engine.__class__.__name__} (自动回退)")
                
            if not VLLM_AVAILABLE:
                engine = create_engine('vllm', config)
                details.append(f"VLLM不可用 → {engine.__class__.__name__} (自动回退)")
                
            return True, "引擎回退机制正常", details
            
        except Exception as e:
            return False, f"引擎回退测试失败: {str(e)}", []
    
    async def test_model_loading(self):
        """测试模型加载功能"""
        try:
            details = []
            
            # 检查是否有可用的模型文件
            model_paths = [
                "./models/qwen3-7b",
                "./models/qwen-7b", 
                "./models/Qwen2.5-7B-Instruct",
                "./models/qwen-0.5b"
            ]
            
            available_model = None
            for model_path in model_paths:
                if os.path.exists(model_path):
                    available_model = model_path
                    break
            
            if not available_model:
                details.append("未找到可用的模型文件")
                details.extend([f"检查路径: {path}" for path in model_paths])
                return False, "未找到模型文件", details
            
            # 获取最优引擎
            engine_type = get_optimal_engine()
            details.append(f"使用引擎: {engine_type}")
            details.append(f"模型路径: {available_model}")
            
            # 创建引擎配置
            config = EngineConfig(
                engine_type=engine_type,
                device_type='auto',
                max_gpu_memory=0.8,
                max_cpu_threads=8,
                max_sequence_length=2048
            )
            
            # 创建并初始化引擎
            self.engine_instance = create_engine(engine_type, config)
            init_success = await self.engine_instance.initialize()
            
            if not init_success:
                return False, "引擎初始化失败", details
            
            details.append("引擎初始化成功")
            
            # 尝试加载模型
            model_name = "qwen-test"
            load_start = time.time()
            load_success = await self.engine_instance.load_model(model_name, available_model)
            load_time = time.time() - load_start
            
            if load_success:
                details.append(f"模型加载成功，耗时: {load_time:.2f}s")
                self.performance_results['model_load_time'] = load_time
                return True, f"模型加载成功 ({load_time:.2f}s)", details
            else:
                return False, "模型加载失败", details
                
        except Exception as e:
            return False, f"模型加载测试失败: {str(e)}", [str(e)]
    
    async def test_token_generation_speed(self):
        """测试 token 生成速度"""
        if not self.engine_instance:
            return False, "引擎未初始化", ["需要先通过模型加载测试"]
        
        try:
            details = []
            
            # 测试文本
            test_prompts = [
                "请简单介绍一下人工智能的发展历史",
                "解释一下深度学习的基本原理",
                "Python 有哪些主要的特点和优势？"
            ]
            
            token_speeds = []
            response_times = []
            
            for i, prompt in enumerate(test_prompts):
                details.append(f"测试 {i+1}: {prompt[:30]}...")
                
                # 创建推理请求
                request = InferenceRequest(
                    model_name="qwen-test",
                    prompt=prompt,
                    max_tokens=100,
                    temperature=0.7,
                    top_p=0.9
                )
                
                # 执行推理
                start_time = time.time()
                try:
                    response = await self.engine_instance.generate(request)
                    end_time = time.time()
                    
                    response_time = end_time - start_time
                    response_times.append(response_time)
                    
                    # 估算 token 数量（简单估计：字符数 / 2）
                    estimated_tokens = len(response.text) // 2
                    token_speed = estimated_tokens / response_time if response_time > 0 else 0
                    token_speeds.append(token_speed)
                    
                    details.append(f"  响应时间: {response_time:.2f}s")
                    details.append(f"  生成文本长度: {len(response.text)} 字符")
                    details.append(f"  估算 tokens: {estimated_tokens}")
                    details.append(f"  Token 速度: {token_speed:.1f} tokens/s")
                    
                except Exception as e:
                    details.append(f"  推理失败: {str(e)}")
                    continue
            
            if token_speeds:
                avg_token_speed = statistics.mean(token_speeds)
                avg_response_time = statistics.mean(response_times)
                
                self.performance_results.update({
                    'avg_token_speed': avg_token_speed,
                    'avg_response_time': avg_response_time,
                    'token_speeds': token_speeds,
                    'response_times': response_times
                })
                
                details.append(f"平均 Token 速度: {avg_token_speed:.1f} tokens/s")
                details.append(f"平均响应时间: {avg_response_time:.2f}s")
                
                return True, f"Token 生成测试完成 (平均 {avg_token_speed:.1f} tokens/s)", details
            else:
                return False, "所有推理请求都失败了", details
                
        except Exception as e:
            return False, f"Token 生成测试失败: {str(e)}", [str(e)]
    
    async def test_concurrent_throughput(self):
        """测试3个并发请求的吞吐速度"""
        if not self.engine_instance:
            return False, "引擎未初始化", ["需要先通过模型加载测试"]
        
        try:
            details = []
            concurrent_requests = 3  # 降低并发数量以避免内存问题
            
            # 并发测试的提示词（减少到三个）
            test_prompts = [
                "什么是机器学习？",
                "Python的特点？",
                "深度学习是什么？"
            ][:concurrent_requests]  # 确保提示词数量与请求数匹配
            
            details.append(f"开始 {concurrent_requests} 个并发请求测试...")
            
            # 创建并发请求
            requests = []
            for i in range(concurrent_requests):
                request = InferenceRequest(
                    model_name="qwen-test",
                    prompt=test_prompts[i],
                    max_tokens=30,  # 减少token数以加快测试
                    temperature=0.7
                )
                requests.append(request)
            
            # 执行并发测试
            start_time = time.time()
            
            async def single_request(req_id, request):
                """单个请求处理"""
                req_start = time.time()
                try:
                    response = await self.engine_instance.generate(request)
                    req_end = time.time()
                    return {
                        'id': req_id,
                        'success': True,
                        'response_time': req_end - req_start,
                        'text_length': len(response.text),
                        'estimated_tokens': len(response.text) // 2
                    }
                except Exception as e:
                    req_end = time.time()
                    return {
                        'id': req_id,
                        'success': False,
                        'response_time': req_end - req_start,
                        'error': str(e)
                    }
            
            # 并发执行，但添加延迟以避免同时启动
            async def delayed_request(delay, req_id, request):
                await asyncio.sleep(delay * 0.1)  # 每个请求间隔100ms
                return await single_request(req_id, request)
            
            tasks = [delayed_request(i, i, req) for i, req in enumerate(requests)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            total_time = time.time() - start_time
            
            # 分析结果
            successful_requests = [r for r in results if isinstance(r, dict) and r.get('success', False)]
            failed_requests = len(results) - len(successful_requests)
            
            if successful_requests:
                response_times = [r['response_time'] for r in successful_requests]
                total_tokens = sum(r['estimated_tokens'] for r in successful_requests)
                
                avg_response_time = statistics.mean(response_times)
                max_response_time = max(response_times)
                min_response_time = min(response_times)
                
                # 计算吞吐量
                requests_per_second = len(successful_requests) / total_time
                tokens_per_second = total_tokens / total_time
                
                self.performance_results.update({
                    'concurrent_total_time': total_time,
                    'concurrent_successful': len(successful_requests),
                    'concurrent_failed': failed_requests,
                    'concurrent_rps': requests_per_second,
                    'concurrent_tps': tokens_per_second,
                    'concurrent_avg_response': avg_response_time,
                    'concurrent_min_response': min_response_time,
                    'concurrent_max_response': max_response_time
                })
                
                details.append(f"总执行时间: {total_time:.2f}s")
                details.append(f"成功请求: {len(successful_requests)}/{concurrent_requests}")
                details.append(f"失败请求: {failed_requests}")
                details.append(f"请求吞吐量: {requests_per_second:.2f} req/s")
                details.append(f"Token 吞吐量: {tokens_per_second:.1f} tokens/s")
                details.append(f"平均响应时间: {avg_response_time:.2f}s")
                details.append(f"最小响应时间: {min_response_time:.2f}s")
                details.append(f"最大响应时间: {max_response_time:.2f}s")
                
                # 详细的每个请求结果
                for i, result in enumerate(successful_requests):
                    details.append(f"请求 {result['id']+1}: {result['response_time']:.2f}s, {result['estimated_tokens']} tokens")
                
                success_rate = len(successful_requests) / concurrent_requests * 100
                return True, f"并发测试完成 ({success_rate:.0f}% 成功率, {requests_per_second:.2f} req/s)", details
            else:
                return False, "所有并发请求都失败了", details
                
        except Exception as e:
            return False, f"并发测试失败: {str(e)}", [str(e)]
    
    def generate_html_report(self, output_file: str = None):
        """生成HTML测试报告"""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"compatibility_report_{timestamp}.html"
            
        html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>跨平台兼容性测试报告</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
        }}
        
        .header p {{
            opacity: 0.9;
            font-size: 1.1rem;
        }}
        
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .summary-card h3 {{
            color: #6c757d;
            font-size: 0.9rem;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .summary-card .value {{
            font-size: 2rem;
            font-weight: bold;
        }}
        
        .summary-card.passed .value {{ color: #28a745; }}
        .summary-card.failed .value {{ color: #dc3545; }}
        .summary-card.total .value {{ color: #007bff; }}
        .summary-card.warnings .value {{ color: #ffc107; }}
        
        .platform-info {{
            padding: 30px;
            background: white;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .platform-info h2 {{
            color: #343a40;
            margin-bottom: 20px;
            font-size: 1.5rem;
        }}
        
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }}
        
        .info-item {{
            display: flex;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        
        .info-label {{
            font-weight: bold;
            color: #6c757d;
            margin-right: 10px;
            min-width: 120px;
        }}
        
        .info-value {{
            color: #343a40;
        }}
        
        .test-results {{
            padding: 30px;
        }}
        
        .test-results h2 {{
            color: #343a40;
            margin-bottom: 20px;
            font-size: 1.5rem;
        }}
        
        .test-item {{
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 10px;
            margin-bottom: 20px;
            overflow: hidden;
            transition: all 0.3s ease;
        }}
        
        .test-item:hover {{
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }}
        
        .test-header {{
            padding: 20px;
            background: #f8f9fa;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
        }}
        
        .test-header.passed {{ background: #d4edda; }}
        .test-header.failed {{ background: #f8d7da; }}
        .test-header.warning {{ background: #fff3cd; }}
        
        .test-name {{
            font-weight: bold;
            color: #343a40;
            font-size: 1.1rem;
        }}
        
        .test-status {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .status-badge {{
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: bold;
            text-transform: uppercase;
        }}
        
        .status-badge.passed {{
            background: #28a745;
            color: white;
        }}
        
        .status-badge.failed {{
            background: #dc3545;
            color: white;
        }}
        
        .status-badge.warning {{
            background: #ffc107;
            color: #343a40;
        }}
        
        .test-duration {{
            color: #6c757d;
            font-size: 0.9rem;
        }}
        
        .test-details {{
            padding: 20px;
            background: white;
            border-top: 1px solid #dee2e6;
            display: none;
        }}
        
        .test-details.show {{
            display: block;
        }}
        
        .test-message {{
            color: #495057;
            margin-bottom: 15px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        
        .detail-list {{
            list-style: none;
        }}
        
        .detail-list li {{
            padding: 8px 0;
            color: #495057;
            border-bottom: 1px solid #f8f9fa;
        }}
        
        .detail-list li:last-child {{
            border-bottom: none;
        }}
        
        .detail-list li:before {{
            content: '▸ ';
            color: #667eea;
            font-weight: bold;
            margin-right: 5px;
        }}
        
        .footer {{
            padding: 20px;
            text-align: center;
            background: #f8f9fa;
            color: #6c757d;
        }}
        
        .success-rate {{
            margin-top: 20px;
            padding: 20px;
            background: white;
            border-radius: 10px;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 30px;
            background: #e9ecef;
            border-radius: 15px;
            overflow: hidden;
            position: relative;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
            transition: width 1s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }}
        
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.8rem;
            }}
            
            .summary {{
                grid-template-columns: 1fr 1fr;
            }}
            
            .info-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 跨平台兼容性测试报告</h1>
            <p>VLLM 推理服务 - 自动化测试结果</p>
        </div>
        
        <div class="summary">
            <div class="summary-card total">
                <h3>测试总数</h3>
                <div class="value">{{total_tests}}</div>
            </div>
            <div class="summary-card passed">
                <h3>通过</h3>
                <div class="value">{{passed}}</div>
            </div>
            <div class="summary-card failed">
                <h3>失败</h3>
                <div class="value">{{failed}}</div>
            </div>
            <div class="summary-card warnings">
                <h3>警告</h3>
                <div class="value">{{warnings}}</div>
            </div>
        </div>
        
        <div class="platform-info">
            <h2>📊 测试环境</h2>
            <div class="info-grid">
                <div class="info-item">
                    <span class="info-label">测试时间:</span>
                    <span class="info-value">{{test_time}}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">操作系统:</span>
                    <span class="info-value">{{platform}}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Python版本:</span>
                    <span class="info-value">{{python_version}}</span>
                </div>
            </div>
            
            <div class="success-rate">
                <h3>成功率</h3>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {{success_rate}}%">
                        {{success_rate}}%
                    </div>
                </div>
            </div>
        </div>
        
        {{performance_section}}
        
        <div class="test-results">
            <h2>🧪 测试结果详情</h2>
            {{test_items}}
        </div>
        
        <div class="footer">
            <p>© 2024 VLLM 跨平台推理服务 | 自动生成的测试报告</p>
        </div>
    </div>
    
    <script>
        // 点击展开/收起测试详情
        document.querySelectorAll('.test-header').forEach(header => {{
            header.addEventListener('click', () => {{
                const details = header.nextElementSibling;
                details.classList.toggle('show');
            }});
        }});
        
        // 自动展开失败的测试
        document.querySelectorAll('.test-header.failed').forEach(header => {{
            const details = header.nextElementSibling;
            details.classList.add('show');
        }});
    </script>
</body>
</html>
        """
        
        # 生成测试项HTML
        test_items_html = ""
        for test in self.test_results:
            status_class = test['status']
            details_html = ""
            
            if test['details']:
                details_html = "<ul class='detail-list'>"
                for detail in test['details']:
                    details_html += f"<li>{detail}</li>"
                details_html += "</ul>"
            
            test_items_html += f"""
            <div class="test-item">
                <div class="test-header {status_class}">
                    <div>
                        <div class="test-name">{test['name']}</div>
                        <div style="color: #6c757d; font-size: 0.9rem; margin-top: 5px;">{test['category']}</div>
                    </div>
                    <div class="test-status">
                        <span class="test-duration">{test['duration']}s</span>
                        <span class="status-badge {status_class}">
                            {'✓ 通过' if test['status'] == 'passed' else '✗ 失败'}
                        </span>
                    </div>
                </div>
                <div class="test-details">
                    <div class="test-message">{test['message']}</div>
                    {details_html}
                </div>
            </div>
            """
        
        # 计算成功率
        success_rate = 0
        if self.summary['total_tests'] > 0:
            success_rate = round((self.summary['passed'] / self.summary['total_tests']) * 100, 1)
        
        # 生成性能结果部分
        performance_section = ""
        if self.performance_results:
            performance_cards = ""
            
            if 'avg_token_speed' in self.performance_results:
                performance_cards += f"""
                <div class="summary-card">
                    <h3>Token 速度</h3>
                    <div class="value" style="color: #17a2b8;">{self.performance_results['avg_token_speed']:.1f}</div>
                    <small>tokens/s</small>
                </div>
                """
            
            if 'avg_response_time' in self.performance_results:
                performance_cards += f"""
                <div class="summary-card">
                    <h3>平均响应时间</h3>
                    <div class="value" style="color: #6610f2;">{self.performance_results['avg_response_time']:.2f}s</div>
                    <small>秒</small>
                </div>
                """
            
            if 'concurrent_rps' in self.performance_results:
                performance_cards += f"""
                <div class="summary-card">
                    <h3>并发吞吐量</h3>
                    <div class="value" style="color: #e83e8c;">{self.performance_results['concurrent_rps']:.2f}</div>
                    <small>req/s</small>
                </div>
                """
            
            if 'model_load_time' in self.performance_results:
                performance_cards += f"""
                <div class="summary-card">
                    <h3>模型加载时间</h3>
                    <div class="value" style="color: #fd7e14;">{self.performance_results['model_load_time']:.2f}s</div>
                    <small>秒</small>
                </div>
                """
            
            if performance_cards:
                performance_section = f"""
                <div class="platform-info">
                    <h2>⚡ 性能指标</h2>
                    <div class="summary">
                        {performance_cards}
                    </div>
                </div>
                """
        
        # 填充模板
        html_content = html_template.format(
            total_tests=self.summary['total_tests'],
            passed=self.summary['passed'],
            failed=self.summary['failed'],
            warnings=self.summary['warnings'],
            test_time=self.summary['test_time'],
            platform=self.summary['platform'],
            python_version=self.summary['python_version'],
            success_rate=success_rate,
            performance_section=performance_section,
            test_items=test_items_html
        )
        
        # 保存文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return output_file
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("=" * 60)
        print("🚀 开始运行跨平台兼容性和性能测试...")
        print("=" * 60)
        
        # 基础功能测试（同步）
        basic_tests = [
            ("平台检测", self.test_platform_detection, "基础功能"),
            ("引擎可用性检查", self.test_engine_availability, "基础功能"),
            ("配置系统验证", self.test_config_validation, "核心功能"),
            ("路径兼容性测试", self.test_path_compatibility, "兼容性"),
            ("引擎回退机制", self.test_engine_fallback, "容错机制"),
        ]
        
        # 性能测试（异步）
        performance_tests = [
            ("模型加载测试", self.test_model_loading, "性能测试"),
            ("Token生成速度测试", self.test_token_generation_speed, "性能测试"),
            ("并发吞吐量测试", self.test_concurrent_throughput, "性能测试"),
        ]
        
        # 运行基础功能测试
        for test_name, test_func, category in basic_tests:
            print(f"\n正在运行: {test_name}...", end=" ")
            result = self.run_test(test_name, test_func, category)
            
            if result['status'] == 'passed':
                print("✅ 通过")
            elif result['status'] == 'failed':
                print("❌ 失败")
            else:
                print("⚠️ 警告")
        
        # 运行性能测试（异步）
        print("\n" + "=" * 60)
        print("🚀 开始性能测试...")
        print("=" * 60)
        
        for test_name, test_func, category in performance_tests:
            print(f"\n正在运行: {test_name}...", end=" ")
            
            # 异步测试需要特殊处理
            try:
                start_time = time.time()
                test_result = await test_func()
                duration = round(time.time() - start_time, 3)
                
                result = {
                    'name': test_name,
                    'category': category,
                    'duration': duration
                }
                
                if isinstance(test_result, tuple):
                    success, message, details = test_result
                    result['status'] = 'passed' if success else 'failed'
                    result['message'] = message
                    result['details'] = details if isinstance(details, list) else [details]
                else:
                    result['status'] = 'passed' if test_result else 'failed'
                    result['message'] = '测试通过' if test_result else '测试失败'
                    result['details'] = []
                
                self.test_results.append(result)
                self.summary['total_tests'] += 1
                
                if result['status'] == 'passed':
                    self.summary['passed'] += 1
                    print("✅ 通过")
                else:
                    self.summary['failed'] += 1
                    print("❌ 失败")
                    
            except Exception as e:
                print(f"❌ 异常: {str(e)}")
                result = {
                    'name': test_name,
                    'category': category,
                    'status': 'failed',
                    'message': str(e),
                    'details': [traceback.format_exc()],
                    'duration': 0
                }
                self.test_results.append(result)
                self.summary['total_tests'] += 1
                self.summary['failed'] += 1
        
        # 生成报告
        print("\n" + "=" * 60)
        print("📊 生成测试报告...")
        report_file = self.generate_html_report()
        
        print(f"\n✅ 测试报告已生成: {report_file}")
        print(f"📈 成功率: {self.summary['passed']}/{self.summary['total_tests']} ({round((self.summary['passed']/self.summary['total_tests']*100) if self.summary['total_tests'] > 0 else 0, 1)}%)")
        
        # 尝试自动打开报告
        try:
            import webbrowser
            webbrowser.open(f'file://{os.path.abspath(report_file)}')
            print("🌐 已在浏览器中打开报告")
        except:
            print(f"💡 请手动打开报告文件查看: {report_file}")
        
        return report_file


async def main():
    """主函数"""
    reporter = CompatibilityTestReporter()
    report_file = await reporter.run_all_tests()
    
    print("\n" + "=" * 60)
    print("✨ 测试完成！")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))