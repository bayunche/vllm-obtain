"""
测试工具和辅助函数
"""

import time
import asyncio
import requests
import json
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager


class TestServiceRunner:
    """测试服务运行器"""
    
    def __init__(self, host="127.0.0.1", port=8001):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.api_base = f"{self.base_url}/v1"
        self.process = None
        self.is_running = False
    
    def wait_for_service(self, timeout=30):
        """等待服务启动"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    self.is_running = True
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        return False
    
    def check_service_health(self):
        """检查服务健康状态"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False


class APITestClient:
    """API测试客户端"""
    
    def __init__(self, base_url="http://127.0.0.1:8001"):
        self.base_url = base_url
        self.api_base = f"{base_url}/v1"
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> Dict[str, Any]:
        """获取模型列表"""
        response = self.session.get(f"{self.api_base}/models")
        response.raise_for_status()
        return response.json()
    
    def chat_completion(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """聊天补全"""
        response = self.session.post(f"{self.api_base}/chat/completions", json=request)
        response.raise_for_status()
        return response.json()
    
    def text_completion(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """文本补全"""
        response = self.session.post(f"{self.api_base}/completions", json=request)
        response.raise_for_status()
        return response.json()
    
    def stream_chat_completion(self, request: Dict[str, Any]):
        """流式聊天补全"""
        request["stream"] = True
        response = self.session.post(
            f"{self.api_base}/chat/completions", 
            json=request, 
            stream=True
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]
                    if data_str.strip() == '[DONE]':
                        break
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        continue


class PerformanceTimer:
    """性能计时器"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """停止计时"""
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        return self.duration
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class ResponseValidator:
    """API响应验证器"""
    
    @staticmethod
    def validate_models_response(response: Dict[str, Any]) -> bool:
        """验证模型列表响应"""
        required_fields = ["object", "data"]
        for field in required_fields:
            if field not in response:
                return False
        
        if response["object"] != "list":
            return False
        
        for model in response["data"]:
            if not all(field in model for field in ["id", "object"]):
                return False
            if model["object"] != "model":
                return False
        
        return True
    
    @staticmethod
    def validate_chat_completion_response(response: Dict[str, Any]) -> bool:
        """验证聊天补全响应"""
        required_fields = ["id", "object", "created", "model", "choices", "usage"]
        for field in required_fields:
            if field not in response:
                return False
        
        if response["object"] != "chat.completion":
            return False
        
        if not response["choices"] or len(response["choices"]) == 0:
            return False
        
        choice = response["choices"][0]
        if not all(field in choice for field in ["message", "finish_reason"]):
            return False
        
        message = choice["message"]
        if not all(field in message for field in ["role", "content"]):
            return False
        
        usage = response["usage"]
        if not all(field in usage for field in ["prompt_tokens", "completion_tokens", "total_tokens"]):
            return False
        
        return True
    
    @staticmethod
    def validate_text_completion_response(response: Dict[str, Any]) -> bool:
        """验证文本补全响应"""
        required_fields = ["id", "object", "created", "model", "choices", "usage"]
        for field in required_fields:
            if field not in response:
                return False
        
        if response["object"] != "text_completion":
            return False
        
        if not response["choices"] or len(response["choices"]) == 0:
            return False
        
        choice = response["choices"][0]
        if "text" not in choice:
            return False
        
        return True


class MockEngine:
    """模拟推理引擎，用于测试"""
    
    def __init__(self, engine_type="mock"):
        self.engine_type = engine_type
        self.is_initialized = False
        self.loaded_models = {}
    
    async def initialize(self):
        """初始化"""
        await asyncio.sleep(0.1)  # 模拟初始化时间
        self.is_initialized = True
        return True
    
    async def load_model(self, model_name: str, model_path: str):
        """加载模型"""
        await asyncio.sleep(0.1)  # 模拟加载时间
        self.loaded_models[model_name] = model_path
        return True
    
    async def unload_model(self, model_name: str):
        """卸载模型"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
        return True
    
    async def generate(self, request):
        """生成文本"""
        await asyncio.sleep(0.1)  # 模拟推理时间
        
        # 模拟响应
        if hasattr(request, 'prompt'):
            # 文本补全
            return {
                "text": f"这是对'{request.prompt}'的模拟回复。",
                "prompt_tokens": len(request.prompt.split()),
                "completion_tokens": 10,
                "total_tokens": len(request.prompt.split()) + 10
            }
        else:
            # 聊天补全
            return {
                "message": {"role": "assistant", "content": "这是一个测试回复。"},
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "total_tokens": 25
            }
    
    def is_model_loaded(self, model_name: str) -> bool:
        """检查模型是否已加载"""
        return model_name in self.loaded_models


def create_test_requests():
    """创建测试请求"""
    return {
        "simple_chat": {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 50
        },
        "complex_chat": {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain AI in simple terms."}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        },
        "simple_completion": {
            "model": "test-model",
            "prompt": "The future of AI is",
            "max_tokens": 30
        }
    }


@asynccontextmanager
async def temporary_service(config=None):
    """临时服务上下文管理器"""
    # 这里可以实现启动临时服务的逻辑
    # 目前先假设服务已经在运行
    yield APITestClient()


def wait_for_condition(condition_func, timeout=30, interval=1):
    """等待条件满足"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if condition_func():
            return True
        time.sleep(interval)
    return False