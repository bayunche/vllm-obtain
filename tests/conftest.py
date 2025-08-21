"""
pytest 配置文件
定义全局测试工具和fixtures
"""

import pytest
import asyncio
import os
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_config, setup_logger
from src.core.model_manager import ModelManager
from src.api.app import create_app


@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config():
    """测试配置"""
    # 设置测试环境变量
    os.environ.setdefault('INFERENCE_ENGINE', 'llama_cpp')  # 使用兼容性最好的引擎
    os.environ.setdefault('HOST', '127.0.0.1')
    os.environ.setdefault('PORT', '8001')  # 避免与开发服务冲突
    os.environ.setdefault('MAX_CONCURRENT_MODELS', '1')
    
    return get_config()


@pytest.fixture
def test_logger():
    """测试日志器"""
    config = get_config()
    return setup_logger(config)


@pytest.fixture
async def model_manager(test_config):
    """测试模型管理器"""
    manager = ModelManager(test_config)
    await manager.initialize()
    yield manager
    # 清理
    await manager.cleanup()


@pytest.fixture
def test_app(test_config):
    """测试Flask应用"""
    app = create_app(test_config)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_chat_request():
    """示例聊天请求"""
    return {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "你是一个有用的助手。"},
            {"role": "user", "content": "请简单说一句话。"}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    }


@pytest.fixture
def sample_completion_request():
    """示例文本补全请求"""
    return {
        "model": "test-model",
        "prompt": "人工智能是",
        "max_tokens": 30,
        "temperature": 0.5
    }


class TestMetrics:
    """测试指标收集器"""
    
    def __init__(self):
        self.response_times = []
        self.request_counts = 0
        self.error_counts = 0
        self.start_time = time.time()
    
    def record_request(self, response_time, success=True):
        """记录请求"""
        self.response_times.append(response_time)
        self.request_counts += 1
        if not success:
            self.error_counts += 1
    
    def get_stats(self):
        """获取统计信息"""
        if not self.response_times:
            return {}
        
        total_time = time.time() - self.start_time
        avg_response_time = sum(self.response_times) / len(self.response_times)
        
        return {
            "total_requests": self.request_counts,
            "successful_requests": self.request_counts - self.error_counts,
            "error_requests": self.error_counts,
            "success_rate": (self.request_counts - self.error_counts) / self.request_counts * 100,
            "avg_response_time": avg_response_time,
            "min_response_time": min(self.response_times),
            "max_response_time": max(self.response_times),
            "throughput": self.request_counts / total_time if total_time > 0 else 0
        }


@pytest.fixture
def test_metrics():
    """测试指标收集器"""
    return TestMetrics()


def pytest_configure(config):
    """pytest 配置钩子"""
    # 添加自定义标记
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "stress: marks tests as stress tests"
    )


def pytest_runtest_setup(item):
    """测试运行前的设置"""
    # 可以在这里添加测试前的准备工作
    pass


def pytest_runtest_teardown(item):
    """测试运行后的清理"""
    # 可以在这里添加测试后的清理工作
    pass